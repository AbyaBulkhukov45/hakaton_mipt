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
    # DuckDB на Windows лучше работает с C:/... (POSIX-слеши), а не backslashes.
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

    parquet_sql = _sql_str(_as_duckdb_path(events_path))
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet({parquet_sql});")

    # DAU
    dau = con.execute(
        """
        SELECT date_trunc('day', event_ts) AS d, COUNT(DISTINCT device_id) AS dau
        FROM events
        GROUP BY 1
        ORDER BY 1
        """
    ).df()

    plt.figure()
    plt.plot(dau["d"], dau["dau"])
    plt.xticks(rotation=45, ha="right")
    plt.title("DAU (unique devices per day)")
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "dau.png", dpi=150)
    plt.close()

    # Top functional
    top_feature = con.execute(
        """
        SELECT feature, COUNT(*) AS cnt
        FROM events
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 20
        """
    ).df()

    plt.figure()
    plt.barh(top_feature["feature"][::-1], top_feature["cnt"][::-1])
    plt.title("Top-20 Functional (by events count)")
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "top_feature.png", dpi=150)
    plt.close()

    # OS split
    os_cnt = con.execute(
        """
        SELECT os, COUNT(DISTINCT device_id) AS devices
        FROM events
        GROUP BY 1
        ORDER BY 2 DESC
        """
    ).df()

    plt.figure()
    plt.bar(os_cnt["os"].astype(str), os_cnt["devices"])
    plt.title("Devices by OS")
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "devices_by_os.png", dpi=150)
    plt.close()

    con.close()
    logger.info("Saved figures to %s", cfg.out_dir.resolve())


if __name__ == "__main__":
    main()
