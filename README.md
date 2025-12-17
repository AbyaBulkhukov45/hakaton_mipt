## Практикум МФТИ: ЖКХ app analytics

### Быстрый запуск (NLP)
1) Положить `Обращения.xlsx` в `data/raw/Обращения.xlsx`
2) `pip install -r requirements.txt`
3) `python -m src.nlp.train_classifier`

Выходные артефакты: `artifacts/nlp/*`





# HAKATON_MIPT — ЖКХ Mobile App Analytics (Case 3)

Цель: система анализа поведения пользователей мобильного приложения ЖКХ для:
- выявления паттернов поведения и сегментов (кластеризация)
- прогнозирования оттока (churn)
- анализа обращений/отзывов (NLP: темы/классификация)
- подготовки отчёта/визуализаций и рекомендаций для бизнес-заказчика

---

## ВАЖНО (NDA / данные)
Сырые данные **не коммитим** в GitHub.  
Данные должны лежать локально в `data/raw/`.

Ожидаемые файлы:
- `data/raw/dataset_new.csv` — события приложения
- `data/raw/словарь_соцдема.csv` — соцдем (device_id → возраст/пол)
- `data/raw/Обращения.xlsx` — обращения/отзывы (тексты) для NLP

---

## 1) Что на выходе должно получиться (артефакты)
После полного запуска вы получите:

### ETL (Parquet)
- `data/processed/events.parquet`
- `data/processed/socdem.parquet`

### EDA (графики + таблички)
Папка: `artifacts/figures/`
- `dau.png` — DAU (устройства/день)
- `devices_by_os.png` — устройства по OS
- `top_feature.png` — топ функционала по событиям
- `retention_d1_d7.png` — retention по когортам (D1/D7)
- `retention_d1_d7_by_cohort.csv`
- `adoption_by_day.png` — доля активных, использовавших ключевые функции
- `adoption_by_day.csv`

### Сегментация поведения
Папка: `artifacts/behavior/`
- `user_clusters.csv` — device_id → cluster
- `cluster_profiles.csv` — профиль кластеров (средние фичи)
- `metrics.json` — silhouette / best_k
- `churn_by_cluster.csv` — churn_rate по сегментам
- `churn_by_cluster.png`

### Churn (прогноз оттока)
Папка: `artifacts/churn/`
- `churn_model.joblib`
- `metrics.json` — ROC-AUC / PR-AUC / churn_rate / rows

### NLP (обращения/отзывы)
Папка: `artifacts/nlp/`
- `text_topic_model.joblib`
- `metrics.json`
- `subtopics.json`
- `misclassified.csv`

### Ноутбуки для защиты/презентации
- `notebooks/01_events_eda.ipynb` — EDA + выводы
- `notebooks/02_models_summary.ipynb` — churn + кластера + NLP + рекомендации

---

## 2) Структура проекта
