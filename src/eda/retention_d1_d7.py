from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


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
    out_dir: Path = Path("artifacts/figures")


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

    # cohort = first_day
    df = con.execute(
        """
        WITH base AS (
            SELECT
                device_id,
                date_trunc('day', event_ts) AS d
            FROM events
        ),
        first_seen AS (
            SELECT device_id, MIN(d) AS cohort_day
            FROM base
            GROUP BY 1
        ),
        flags AS (
            SELECT
                f.cohort_day,
                f.device_id,
                MAX(CASE WHEN b.d = f.cohort_day + INTERVAL '1 day' THEN 1 ELSE 0 END) AS d1,
                MAX(CASE WHEN b.d = f.cohort_day + INTERVAL '7 day' THEN 1 ELSE 0 END) AS d7
            FROM first_seen f
            LEFT JOIN base b ON b.device_id = f.device_id
            GROUP BY 1, 2
        )
        SELECT
            cohort_day,
            COUNT(*) AS cohort_size,
            AVG(d1) AS retention_d1,
            AVG(d7) AS retention_d7
        FROM flags
        GROUP BY 1
        ORDER BY 1
        """
    ).df()

    df.to_csv(cfg.out_dir / "retention_d1_d7_by_cohort.csv", index=False, encoding="utf-8")

    # plot
    plt.figure()
    plt.plot(df["cohort_day"], df["retention_d1"], label="D1")
    plt.plot(df["cohort_day"], df["retention_d7"], label="D7")
    plt.xticks(rotation=45, ha="right")
    plt.title("Retention by cohort (D1/D7)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "retention_d1_d7.png", dpi=150)
    plt.close()

    con.close()
    logger.info("Saved: %s", (cfg.out_dir / "retention_d1_d7.png").resolve())


if __name__ == "__main__":
    main()
