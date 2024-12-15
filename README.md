# Conv CP Decomposition

Этот проект является реализацией данной статьи: [arxiv](https://arxiv.org/abs/1701.07148) в рамках курса "Вычислительная линейная алгебра" AI Masters.

---
Авторы: Загайнов Никита, Агаджанов Хусейин, Кривенко Павел

## Инструкция по использованию
1. Клонировать репозиторий
```bash
git clone https://github.com/SqVaD-NLA2024/conv-cp-decomposition.git
cd conv-cp-decomposition
```

2. Установить как библиотеку в `dev`-режиме
```bash
pip install -e .
```

3. Чтобы разложить сверточный слой модели, используйте следующий код:
```python
from conv_cp import CPConv2d

cp_layer = CPConv2d(reference_layer, rank=10) # reference_layer - слой, который нужно разложить, rank - ранг разложения
```

