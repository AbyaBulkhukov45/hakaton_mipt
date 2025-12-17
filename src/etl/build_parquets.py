from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


def _sql_str(value: str) -> str:
    """Return SQL single-quoted string literal with escaping."""
    return "'" + value.replace("'", "''") + "'"


def _as_duckdb_path(path: Path) -> str:
    # DuckDB отлично переваривает C:/... (POSIX-слеши), а не Windows backslashes.
    return path.resolve().as_posix()


def _glob_many(base_dir: Path, patterns: Iterable[str]) -> Iterator[Path]:
    for pat in patterns:
        yield from base_dir.glob(pat)


@dataclass(frozen=True)
class EtlConfig:
    # Events input/output
    input_path: Path | None = None
    output_path: Path | None = None

    # Socdem input/output
    socdem_path: Path | None = None
    socdem_output_path: Path | None = None

    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolved_input_path(self) -> Path:
        """
        Resolve events CSV:
        1) EVENTS_CSV_PATH env
        2) EtlConfig.input_path
        3) latest *.csv/*.scv in data/raw
        """
        env_path = os.getenv("EVENTS_CSV_PATH")
        if env_path:
            p = Path(env_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"EVENTS_CSV_PATH points to missing file: {p}")
            return p

        if self.input_path is not None:
            p = self.input_path.expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Input file not found: {p}")
            return p

        raw_dir = self.project_root() / "data" / "raw"
        candidates = list(_glob_many(raw_dir, ["*.csv", "*.scv"]))
        if not candidates:
            raise FileNotFoundError(
                f"No input files found in {raw_dir} (expected *.csv or *.scv). "
                "Provide EtlConfig(input_path=...) or set EVENTS_CSV_PATH."
            )
        return max(candidates, key=lambda x: x.stat().st_mtime)

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path.expanduser()
        return self.project_root() / "data" / "processed" / "events.parquet"

    def resolved_socdem_path(self) -> Path:
        """
        Resolve socdem CSV:
        1) SOCDEM_CSV_PATH env
        2) EtlConfig.socdem_path
        3) data/raw/словарь_соцдема.csv
        """
        env_path = os.getenv("SOCDEM_CSV_PATH")
        if env_path:
            p = Path(env_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"SOCDEM_CSV_PATH points to missing file: {p}")
            return p

        if self.socdem_path is not None:
            p = self.socdem_path.expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Socdem file not found: {p}")
            return p

        p = self.project_root() / "data" / "raw" / "словарь_соцдема.csv"
        if not p.is_file():
            raise FileNotFoundError(
                f"Socdem file not found: {p}. Put it there or set SOCDEM_CSV_PATH."
            )
        return p

    def resolved_socdem_output_path(self) -> Path:
        if self.socdem_output_path is not None:
            return self.socdem_output_path.expanduser()
        return self.project_root() / "data" / "processed" / "socdem.parquet"


def build_events_parquet(cfg: EtlConfig) -> Path:
    """
    Build events.parquet from CSV.

    Timestamp normalization example:
      2025-10-06T17:08:06+03:00[Europe/Moscow]
    -> remove bracketed zone, replace 'T' with space
    -> try_cast(... AS TIMESTAMPTZ) so bad rows don't crash the run
    """
    import duckdb  # local import

    input_path = cfg.resolved_input_path()
    output_path = cfg.resolved_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Events input: %s", input_path)
    logger.info("Events output: %s", output_path)

    input_sql = _sql_str(_as_duckdb_path(input_path))
    output_sql = _sql_str(_as_duckdb_path(output_path))

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=4;")

        con.execute(
            rf"""
            CREATE OR REPLACE TABLE events_norm AS
            WITH src AS (
                SELECT *
                FROM read_csv_auto(
                    {input_sql},
                    header=true,
                    all_varchar=true,
                    ignore_errors=true
                )
            )
            SELECT
                try_cast(
                    replace(
                        regexp_replace("Дата и время события", '\[.*\]$', ''),
                        'T',
                        ' '
                    ) AS TIMESTAMPTZ
                ) AS event_ts,
                nullif(trim("Экран"), '') AS screen,
                nullif(trim("Функционал"), '') AS feature,
                nullif(trim("Действие"), '') AS action,
                try_cast(nullif(trim("Идентификатор устройства"), '') AS BIGINT) AS device_id,
                try_cast(nullif(trim("Номер сессии в рамках устройства"), '') AS BIGINT) AS device_session_id,
                nullif(trim("Производитель устройства"), '') AS device_vendor,
                nullif(trim("Модель устройства"), '') AS device_model,
                nullif(trim("Тип устройства"), '') AS device_type,
                nullif(trim("ОС"), '') AS os
            FROM src;
            """
        )

        # Вырежем строки с битым временем, остальное оставим как есть.
        con.execute(
            rf"""
            COPY (
                SELECT DISTINCT *
                FROM events_norm
                WHERE event_ts IS NOT NULL
                  AND device_id IS NOT NULL
            )
            TO {output_sql}
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        cnt = con.execute(
            "SELECT COUNT(*) FROM events_norm WHERE event_ts IS NOT NULL AND device_id IS NOT NULL;"
        ).fetchone()[0]
        logger.info("Events rows saved: %s", cnt)

        return output_path
    finally:
        con.close()


def build_socdem_parquet(cfg: EtlConfig) -> Path:
    """Build socdem.parquet from словарь_соцдема.csv."""
    import duckdb  # local import

    input_path = cfg.resolved_socdem_path()
    output_path = cfg.resolved_socdem_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Socdem input: %s", input_path)
    logger.info("Socdem output: %s", output_path)

    input_sql = _sql_str(_as_duckdb_path(input_path))
    output_sql = _sql_str(_as_duckdb_path(output_path))

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=2;")

        con.execute(
            rf"""
            CREATE OR REPLACE TABLE socdem_norm AS
            WITH src AS (
                SELECT *
                FROM read_csv_auto(
                    {input_sql},
                    header=true,
                    all_varchar=true,
                    ignore_errors=true
                )
            )
            SELECT
                try_cast(nullif(trim(number), '') AS BIGINT) AS device_id,
                try_cast(nullif(trim(age_back), '') AS INTEGER) AS age,
                nullif(trim(gender), '') AS gender
            FROM src;
            """
        )

        con.execute(
            rf"""
            COPY (
                SELECT DISTINCT *
                FROM socdem_norm
                WHERE device_id IS NOT NULL
            )
            TO {output_sql}
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        cnt = con.execute("SELECT COUNT(*) FROM socdem_norm WHERE device_id IS NOT NULL;").fetchone()[0]
        logger.info("Socdem rows saved: %s", cnt)

        return output_path
    finally:
        con.close()


def build_parquet(cfg: EtlConfig) -> Path:
    """
    Backward-compatible alias: returns events.parquet path.
    (socdem сохраняется отдельной функцией)
    """
    return build_events_parquet(cfg)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = EtlConfig()
    events_out = build_events_parquet(cfg)
    socdem_out = build_socdem_parquet(cfg)

    logger.info("Done events: %s", events_out)
    logger.info("Done socdem: %s", socdem_out)


if __name__ == "__main__":
    main()
