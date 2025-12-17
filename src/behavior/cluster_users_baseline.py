from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

# Чтобы joblib/sklearn не пытались использовать loky (и не дергали wmic на Windows)
os.environ.setdefault("JOBLIB_START_METHOD", "threading")


def _sql_str(value: str) -> str:
    """Return SQL single-quoted string literal with escaping."""
    return "'" + value.replace("'", "''") + "'"


def _as_duckdb_path(path: Path) -> str:
    # DuckDB на Windows лучше воспринимает C:/... (POSIX-слеши), а не backslashes.
    return path.resolve().as_posix()


@dataclass(frozen=True)
class Config:
    events_parquet: Path = Path("data/processed/events.parquet")
    out_dir: Path = Path("artifacts/behavior")
    k_min: int = 3
    k_max: int = 8
    random_state: int = 42

    # Критично для скорости: silhouette O(n^2), поэтому считаем на сэмпле
    silhouette_sample_size: int = 5000  # поставь 2000..10000
    silhouette_n_jobs: int = 1  # 1 = без loky/процессов


def _silhouette_on_sample(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
    n_jobs: int,
) -> float:
    """
    Быстрый silhouette: считаем на случайном подмножестве.
    """
    n = x_scaled.shape[0]
    if n < 3:
        raise ValueError("Not enough samples for silhouette.")

    # silhouette требует минимум 2 кластера и не должно быть кластеров размера 1 в сэмпле
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        raise ValueError("Need at least 2 clusters for silhouette.")
    if (counts < 2).any():
        # на полном наборе уже есть кластера по 1 элементу — silhouette будет нестабилен
        raise ValueError("Some clusters have <2 samples; silhouette undefined.")

    if n <= sample_size:
        return float(silhouette_score(x_scaled, labels, metric="euclidean", n_jobs=n_jobs))

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)

    x_s = x_scaled[idx]
    y_s = labels[idx]

    # проверим, что в сэмпле не выпал кластер до размера 1
    _, c_s = np.unique(y_s, return_counts=True)
    if (c_s < 2).any():
        # пересэмплим пару раз и если не получится — откажемся
        for _ in range(5):
            idx = rng.choice(n, size=sample_size, replace=False)
            x_s = x_scaled[idx]
            y_s = labels[idx]
            _, c_s = np.unique(y_s, return_counts=True)
            if not (c_s < 2).any():
                break
        else:
            raise ValueError("Failed to sample without singleton clusters for silhouette.")

    return float(silhouette_score(x_s, y_s, metric="euclidean", n_jobs=n_jobs))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    events_path = cfg.events_parquet.resolve()
    if not events_path.exists():
        raise FileNotFoundError(f"Parquet not found: {events_path}")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")

    events_sql = _sql_str(_as_duckdb_path(events_path))
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet({events_sql});")

    df = con.execute(
        """
        SELECT
            device_id,
            COUNT(*) AS events_cnt,
            COUNT(DISTINCT date_trunc('day', event_ts)) AS active_days,
            COUNT(DISTINCT device_session_id) AS sessions_cnt,
            COUNT(DISTINCT screen) AS screens_cnt,
            COUNT(DISTINCT feature) AS features_cnt,
            SUM(CASE WHEN lower(feature) LIKE '%оплат%' THEN 1 ELSE 0 END) AS pay_events,
            SUM(CASE WHEN lower(feature) LIKE '%заявк%' THEN 1 ELSE 0 END) AS ticket_events,
            SUM(CASE WHEN lower(feature) LIKE '%показан%' THEN 1 ELSE 0 END) AS meter_events
        FROM events
        GROUP BY 1
        """
    ).df()

    con.close()

    if df.empty:
        raise ValueError(
            "No rows returned from events. Check events.parquet and required columns "
            "(device_id, event_ts, device_session_id, screen, feature)."
        )

    x = df.drop(columns=["device_id"]).astype(float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32, copy=False)

    best_k: int | None = None
    best_score = float("-inf")
    scores: dict[int, float] = {}

    n = x_scaled.shape[0]
    sample_size = min(cfg.silhouette_sample_size, n)

    logger.info("Users=%d; silhouette sample_size=%d; k_range=[%d..%d]", n, sample_size, cfg.k_min, cfg.k_max)

    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
        labels = km.fit_predict(x_scaled)

        try:
            score = _silhouette_on_sample(
                x_scaled=x_scaled,
                labels=labels,
                sample_size=sample_size,
                random_state=cfg.random_state,
                n_jobs=cfg.silhouette_n_jobs,
            )
        except ValueError as e:
            logger.warning("k=%d: silhouette skipped (%s)", k, e)
            continue

        scores[k] = score
        logger.info("k=%d silhouette=%.6f", k, score)

        if score > best_score:
            best_k = k
            best_score = score

    # Если silhouette не получился ни разу — берем дефолтный k (середина диапазона)
    if best_k is None:
        best_k = int(round((cfg.k_min + cfg.k_max) / 2))
        best_score = float("nan")
        logger.warning("Silhouette failed for all k; fallback best_k=%d", best_k)

    final = KMeans(n_clusters=best_k, random_state=cfg.random_state, n_init=10)
    df["cluster"] = final.fit_predict(x_scaled)

    profile = df.groupby("cluster").mean(numeric_only=True).reset_index()

    df.to_csv(cfg.out_dir / "user_clusters.csv", index=False, encoding="utf-8")
    profile.to_csv(cfg.out_dir / "cluster_profiles.csv", index=False, encoding="utf-8")
    (cfg.out_dir / "metrics.json").write_text(
        json.dumps({"best_k": best_k, "silhouette": best_score, "all_scores": scores}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved behavior artifacts to %s", cfg.out_dir.resolve())


if __name__ == "__main__":
    main()
