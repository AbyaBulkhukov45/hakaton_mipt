from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def _sql_str(value: str) -> str:
    """Return SQL single-quoted string literal with escaping."""
    return "'" + value.replace("'", "''") + "'"


def _as_duckdb_path(path: Path) -> str:
    # DuckDB на Windows лучше воспринимает C:/... (POSIX-слеши), а не backslashes.
    return path.resolve().as_posix()


@dataclass(frozen=True)
class Config:
    events_parquet: Path = Path("data/processed/events.parquet")
    socdem_parquet: Path = Path("data/processed/socdem.parquet")
    out_dir: Path = Path("artifacts/churn")
    feature_days: int = 28
    label_days: int = 14
    test_size: float = 0.2
    random_state: int = 42


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    events_path = cfg.events_parquet.resolve()
    socdem_path = cfg.socdem_parquet.resolve()

    if not events_path.exists():
        raise FileNotFoundError(f"Parquet not found: {events_path}")
    if not socdem_path.exists():
        raise FileNotFoundError(f"Parquet not found: {socdem_path}")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")

    events_sql = _sql_str(_as_duckdb_path(events_path))
    socdem_sql = _sql_str(_as_duckdb_path(socdem_path))

    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet({events_sql});")
    con.execute(f"CREATE OR REPLACE VIEW socdem AS SELECT * FROM read_parquet({socdem_sql});")

    max_ts = con.execute("SELECT MAX(event_ts) FROM events;").fetchone()[0]
    logger.info("max_ts=%s", max_ts)

    df = con.execute(
        f"""
        WITH params AS (
            SELECT
                TIMESTAMPTZ '{max_ts}' AS max_ts,
                (TIMESTAMPTZ '{max_ts}' - INTERVAL '{cfg.label_days} days') AS cutoff,
                (TIMESTAMPTZ '{max_ts}' - INTERVAL '{cfg.label_days + cfg.feature_days} days') AS feature_start
        ),
        base AS (
            SELECT
                e.device_id,
                COUNT(*) AS events_cnt,
                COUNT(DISTINCT date_trunc('day', e.event_ts)) AS active_days,
                COUNT(DISTINCT e.device_session_id) AS sessions_cnt,
                COUNT(DISTINCT e.screen) AS screens_cnt,
                COUNT(DISTINCT e.feature) AS features_cnt,
                SUM(CASE WHEN lower(e.feature) LIKE '%оплат%' THEN 1 ELSE 0 END) AS pay_events,
                SUM(CASE WHEN lower(e.feature) LIKE '%заявк%' THEN 1 ELSE 0 END) AS ticket_events,
                SUM(CASE WHEN lower(e.feature) LIKE '%показан%' THEN 1 ELSE 0 END) AS meter_events
            FROM events e, params p
            WHERE e.event_ts >= p.feature_start AND e.event_ts < p.cutoff
            GROUP BY 1
        ),
        label AS (
            SELECT
                device_id,
                CASE WHEN MAX(event_ts) >= (SELECT cutoff FROM params) THEN 0 ELSE 1 END AS churn
            FROM events
            GROUP BY 1
        )
        SELECT
            b.*,
            l.churn,
            s.age,
            s.gender
        FROM base b
        JOIN label l USING(device_id)
        LEFT JOIN socdem s USING(device_id);
        """
    ).df()

    if df.empty:
        raise ValueError("Feature dataset is empty. Check event_ts range and parquet contents.")

    df["age"] = df["age"].fillna(df["age"].median())
    df["gender"] = df["gender"].fillna("U")
    df = pd.get_dummies(df, columns=["gender"], drop_first=False)

    y = df["churn"].astype(int)
    x = df.drop(columns=["churn"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
    }

    joblib.dump(model, cfg.out_dir / "churn_model.joblib")
    (cfg.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    con.close()
    logger.info("Saved churn artifacts to %s", cfg.out_dir.resolve())
    logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()
