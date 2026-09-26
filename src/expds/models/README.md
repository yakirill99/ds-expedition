# Модуль models

**Владелец:** @DinislamFromUfa

Сегментационные модели для стека 1 (растровый baseline).

## Вход

- `torch.Tensor` формы `(B, C, H, W)`
- `C` — количество каналов (рельеф + оптика + геофизика).  
  Сейчас в коде стоит заглушка `in_channels=10`, реальное значение уточним, когда будет готов модуль `features`.

## Выход

- Логиты формы `(B, 1, H, W)`
- Для получения вероятностей используй метод `.predict()` → `(B, 1, H, W)` в диапазоне `[0, 1]`

## Как использовать

```python
from expds.models import UNetSegmenter, DiceFocalLoss

model = UNetSegmenter(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=10,
    classes=1,
)

criterion = DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)

# Обучение
logits = model(batch)                # (B, 1, H, W)
loss = criterion(logits, masks)

# Инференс
probs = model.predict(batch)         # (B, 1, H, W)
