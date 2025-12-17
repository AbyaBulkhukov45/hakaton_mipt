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

    df = con.execute(
        """
        WITH daily AS (
            SELECT
                date_trunc('day', event_ts) AS d,
                device_id,
                MAX(CASE WHEN lower(feature) LIKE '%оплат%' THEN 1 ELSE 0 END) AS used_pay,
                MAX(CASE WHEN lower(feature) LIKE '%заявк%' THEN 1 ELSE 0 END) AS used_ticket,
                MAX(CASE WHEN lower(feature) LIKE '%показан%' THEN 1 ELSE 0 END) AS used_meter,
                1 AS used_any
            FROM events
            GROUP BY 1, 2
        )
        SELECT
            d,
            AVG(used_pay) AS pay_adoption,
            AVG(used_ticket) AS ticket_adoption,
            AVG(used_meter) AS meter_adoption
        FROM daily
        GROUP BY 1
        ORDER BY 1
        """
    ).df()

    df.to_csv(cfg.out_dir / "adoption_by_day.csv", index=False, encoding="utf-8")

    plt.figure()
    plt.plot(df["d"], df["pay_adoption"], label="Pay")
    plt.plot(df["d"], df["ticket_adoption"], label="Tickets")
    plt.plot(df["d"], df["meter_adoption"], label="Meters")
    plt.xticks(rotation=45, ha="right")
    plt.title("Daily adoption of key features (share of active devices)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "adoption_by_day.png", dpi=150)
    plt.close()

    con.close()
    logger.info("Saved: %s", (cfg.out_dir / "adoption_by_day.png").resolve())


if __name__ == "__main__":
    main()
