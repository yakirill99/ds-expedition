"""predict: точка входа. Заглушка — заменить на реальный пайплайн."""
import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print("predict:", cfg)


if __name__ == "__main__":
    main()
