## Практикум МФТИ: ЖКХ app analytics

### Быстрый запуск (NLP)
1) Положить `Обращения.xlsx` в `data/raw/Обращения.xlsx`
2) `pip install -r requirements.txt`
3) `python -m src.nlp.train_classifier`

Выходные артефакты: `artifacts/nlp/*`






Цель: построить систему анализа поведения пользователей мобильного приложения ЖКХ, чтобы:
- выявить паттерны поведения и сегменты пользователей (кластеризация);
- спрогнозировать отток (churn);
- проанализировать обращения/отзывы пользователей (NLP: классификация, темы, проблемные зоны);
- подготовить отчёт/визуализации и рекомендации для бизнес-заказчика + презентацию.

---

## ВАЖНО (NDA / данные)
Сырые данные **НЕ коммитим** в GitHub.

Ожидается, что данные лежат локально в `data/raw/`:

- `data/raw/dataset_new.csv` — события приложения (или любой `*.csv/*.scv` с теми же колонками)
- `data/raw/словарь_соцдема.csv` — соцдем (device_id → возраст/пол)
- `data/raw/Обращения.xlsx` — обращения/отзывы (тексты) для NLP

Если файлы лежат не в `data/raw/`, можно указать:
- `EVENTS_CSV_PATH` — путь до файла с событиями
- `SOCDEM_CSV_PATH` — путь до соцдемы

---

## 1) Что получается на выходе (артефакты)

После полного запуска пайплайна должны появиться:

### 1.1 ETL (Parquet)
Папка: `data/processed/`
- `events.parquet`
- `socdem.parquet`

### 1.2 EDA (графики + таблицы)
Папка: `artifacts/figures/`
- `dau.png` — DAU (устройства/день)
- `devices_by_os.png` — устройства по OS
- `top_feature.png` — топ функционала по событиям
- `retention_d1_d7.png` — retention по когортам (D1/D7)
- `retention_d1_d7_by_cohort.csv`
- `adoption_by_day.png` — доля активных, использовавших ключевые функции
- `adoption_by_day.csv`

### 1.3 Сегментация поведения (Behavior clustering)
Папка: `artifacts/behavior/`
- `user_clusters.csv` — device_id → cluster
- `cluster_profiles.csv` — профиль кластеров (средние фичи)
- `metrics.json` — silhouette / best_k
- `churn_by_cluster.csv` — churn_rate по сегментам
- `churn_by_cluster.png`

### 1.4 Churn (прогноз оттока)
Папка: `artifacts/churn/`
- `churn_model.joblib` — baseline модель
- `metrics.json` — ROC-AUC / PR-AUC / churn_rate / rows

### 1.5 NLP (обращения/отзывы)
Папка: `artifacts/nlp/`
- `text_topic_model.joblib`
- `metrics.json`
- `subtopics.json` — выделенные проблемные подтемы
- `misclassified.csv` — ошибки классификатора (для контроля качества)

### 1.6 Ноутбуки для защиты/презентации
Папка: `notebooks/`
- `01_events_eda.ipynb` — EDA + выводы
- `02_models_summary.ipynb` — churn + кластера + NLP + рекомендации

---

## 2) Структура проекта

```text
hakaton_mipt/
  artifacts/
    figures/
    behavior/
    churn/
    nlp/
  data/
    raw/                # (NDA) исходные файлы локально
    processed/          # parquet после ETL
  notebooks/
    01_events_eda.ipynb
    02_models_summary.ipynb
  src/
    etl/
      build_parquets.py
    eda/
      events_eda_quick.py
      retention_d1_d7.py
      adoption_key_features.py
    behavior/
      cluster_users_baseline.py
      cluster_churn_report.py
    churn/
      train_churn_baseline.py
    nlp/
      train_classifier.py
  requirements.txt
  .gitignore
  README.md
