"""Read-only coverage audit on the native DTM grid; does not change metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from expds.features.grid import Grid
from expds.io.geojson import GEOM_POLYGON, read_labels
from expds.pipeline_config import load_pipeline_config
from expds.targets.rasterize import points_in_rings


def read_native(path):
    with rasterio.open(path) as ds:
        t = ds.transform
        if ds.crs is None:
            raise ValueError(f"{path}: CRS missing")
        if not (
            t.a > 0
            and t.e < 0
            and abs(t.b) < 1e-12
            and abs(t.d) < 1e-12
            and np.isclose(t.a, -t.e, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(f"{path}: expected north-up square pixels, got {t}")
        a = ds.read(1, masked=True)
        valid = np.flipud(~np.ma.getmaskarray(a) & np.isfinite(a.data)).copy()
        grid = Grid(
            crs=str(ds.crs),
            pixel_size=t.a,
            x_min=t.c,
            y_min=t.f + t.e * ds.height,
            width=ds.width,
            height=ds.height,
        )
        key = (ds.crs, ds.shape, tuple(ds.transform))
        return grid, valid, key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/dataset_real.yaml"))
    p.add_argument("--out", type=Path, default=Path("runs/diagnostics/coverage_objects.csv"))
    p.add_argument("--radius", type=float, default=7.0)
    args = p.parse_args()
    if args.radius < 0:
        p.error("radius must be nonnegative")
    cfg = load_pipeline_config(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "site",
        "label_index",
        "class",
        "status",
        "area_in_grid_px",
        "bbox_crosses_grid",
        "cov_dtm_in_grid",
        "cov_union_in_grid",
        "cov_reachable_in_grid",
        "b3_gt_holes",
        "mean_z_status",
        "meta_status",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for site in sorted(cfg.data.sites_dir.iterdir()):
            if not site.is_dir():
                continue
            dt, labels = site / "layers/dtm.tif", site / cfg.data.labels_file
            if not dt.exists() or not labels.exists():
                writer.writerow(dict(site=site.name, status="missing_dtm_or_labels"))
                continue
            grid, valid, key = read_native(dt)
            meta_status = "absent"
            meta_path = site / "samples/sample_meta.json"
            if meta_path.exists():
                meta = Grid(**json.loads(meta_path.read_text(encoding="utf-8"))["grid"])
                from pyproj import CRS

                same = (
                    CRS(meta.crs) == CRS(grid.crs)
                    and meta.shape == grid.shape
                    and np.allclose(
                        [meta.x_min, meta.y_min, meta.pixel_size],
                        [grid.x_min, grid.y_min, grid.pixel_size],
                        rtol=0,
                        atol=1e-8,
                    )
                )
                meta_status = "matches_dtm" if same else "MISMATCH_NATIVE_DTM_USED"
            vz = np.zeros_like(valid)
            mean_status = "missing"
            mz = site / "layers/mean_z.tif"
            if mz.exists():
                _, candidate, mz_key = read_native(mz)
                if mz_key == key:
                    vz = candidate
                    mean_status = "aligned"
                else:
                    mean_status = "GRID_MISMATCH_NOT_USED"
            reach = valid.copy()
            if valid.any() and not valid.all():
                reach |= ndimage.distance_transform_edt(~valid) <= args.radius
            summary = []
            for index, f in enumerate(read_labels(labels, grid, src_crs=cfg.data.labels_crs)):
                rec = dict(
                    site=site.name,
                    label_index=index,
                    **{"class": f.cls},
                    mean_z_status=mean_status,
                    meta_status=meta_status,
                )
                if f.cls not in cfg.targets.classes:
                    rec["status"] = "unknown_class"
                elif f.geom_type != GEOM_POLYGON:
                    rec["status"] = "not_polygon"
                elif not f.pixel or len(f.pixel[0]) < 3:
                    rec["status"] = "empty_geometry"
                else:
                    ext = np.asarray(f.pixel[0])
                    if not np.isfinite(ext).all():
                        rec["status"] = "nonfinite_geometry"
                        writer.writerow(rec)
                        continue
                    lo, hi = ext.min(axis=0), ext.max(axis=0)
                    crosses = bool(
                        lo[0] < -0.5
                        or lo[1] < -0.5
                        or hi[0] > grid.width - 0.5
                        or hi[1] > grid.height - 0.5
                    )
                    rec["bbox_crosses_grid"] = crosses
                    c0, r0 = np.maximum(np.floor(lo), 0).astype(int)
                    c1, r1 = np.minimum(np.ceil(hi), [grid.width - 1, grid.height - 1]).astype(int)
                    if c0 > c1 or r0 > r1:
                        rec.update(status="outside_grid", area_in_grid_px=0)
                    else:
                        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
                        mask = points_in_rings(cc, rr, f.pixel)
                        area = int(mask.sum())
                        rec["area_in_grid_px"] = area
                        if not area:
                            rec["status"] = "empty_rasterization"
                        else:
                            sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
                            vd, vm = valid[sl], vz[sl]
                            holes = mask & ~vd
                            cov = float(vd[mask].mean())
                            rec.update(
                                status="rasterized",
                                cov_dtm_in_grid=cov,
                                cov_reachable_in_grid=float(reach[sl][mask].mean()),
                            )
                            if mean_status == "aligned":
                                rec["cov_union_in_grid"] = float((vd | vm)[mask].mean())
                                if holes.any():
                                    rec["b3_gt_holes"] = float(vm[holes].mean())
                            summary.append(cov)
                writer.writerow(rec)
            a = np.asarray(summary)
            print(
                site.name,
                f"rasterized={len(a)}",
                meta_status,
                mean_status,
                f"mean={a.mean():.1%} lt30={(a < 0.3).sum()}" if len(a) else "",
                flush=True,
            )
    print(args.out)


if __name__ == "__main__":
    main()
