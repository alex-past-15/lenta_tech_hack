# Распознавание ценников на видео из торгового зала

Решение для хакатона Lenta Tech Life Hack (первый тур): из видео торгового зала получаем структурированную таблицу товаров и цен.

```
видео → выборка кадров → VLM (читает ценники, возвращает JSON) → склейка по кадрам → CSV / JSON
```

| Этап | Модуль | Что делает |
|---|---|---|
| Кадры | `src/pricetags/video.py` | прореживает видео до `sample_fps`, отбрасывает размытые кадры по дисперсии лапласиана |
| Чтение ценников | `src/pricetags/vlm.py` | VLM получает кадр и возвращает JSON: название, цена, старая цена, цена за единицу, штрихкод |
| Агрегация | `src/pricetags/aggregate.py` | один ценник виден на многих кадрах: склеиваем по штрихкоду или названию и голосуем по полям |
| Метрики | `src/pricetags/metrics.py` | CER названия, точность цены, Tag P/R/F1 по уникальным ценникам, Full Match |

## Структура

```
configs/default.yaml     параметры всех этапов
data/videos/             входные видео (сюда положить данные)
data/annotations/        эталонная разметка для оценки
scripts/run.py           прогон пайплайна
scripts/evaluate.py      подсчёт метрик
```

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. положить видео в data/videos
# 2. поднять VLM с OpenAI-совместимым API (например, vLLM) и указать base_url и model в configs/default.yaml
#    для облачного API задать ключ: export VLM_API_KEY=...
python scripts/run.py
python scripts/evaluate.py     # если есть разметка в data/annotations
```

Результат для каждого видео: `outputs/<video>.csv` и `outputs/<video>.json` с полями
`name, price, old_price, unit_price, unit, barcode, conf, n_obs, first_ts, last_ts`.

## Результаты решения на хакатоне

| Метрика | Значение |
|---|---|
| CER названия | 0.15 |
| Price accuracy (matched) | 0.89 |
| Tag Precision / Recall | 0.91 / 0.92 |
| **Tag F1** | **0.91** |
| Full Match (e2e) | 0.81 |

Tag P/R/F1 считаются по уникальным ценникам в видео, Full Match — доля ценников, у которых верны цена, старая цена и штрихкод. Код: `src/pricetags/metrics.py`.
