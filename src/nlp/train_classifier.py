from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger("train_classifier")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    input_path: Path
    artifacts_dir: Path
    text_col: Optional[str]
    target_col: Optional[str]
    test_size: float
    random_state: int
    max_features: int
    min_df: int
    ngram_max: int
    max_iter: int
    min_class_count: int
    merge_rare_to_other: bool


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _clean_text(value: object) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _infer_text_column(df: pd.DataFrame) -> str:
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("Не найдено ни одного текстового (object) столбца для text_col.")

    best_col: Optional[str] = None
    best_score = -1.0

    for col in object_cols:
        series = df[col].astype(str).fillna("")
        avg_len = series.map(len).mean()
        if avg_len > best_score:
            best_score = float(avg_len)
            best_col = col

    if best_col is None:
        raise ValueError("Не удалось определить text_col автоматически.")
    return best_col


def _infer_target_column(df: pd.DataFrame, text_col: str) -> str:
    candidates = [c for c in df.columns if c != text_col]
    if not candidates:
        raise ValueError("Не найдено кандидатов на target_col (кроме text_col).")

    best_col: Optional[str] = None
    best_score = float("inf")
    n_rows = len(df)

    for col in candidates:
        nunique = int(df[col].nunique(dropna=True))
        if nunique <= 1:
            continue
        ratio = nunique / max(n_rows, 1)
        if ratio < best_score:
            best_score = ratio
            best_col = col

    if best_col is None:
        raise ValueError(
            "Не удалось автоматически определить target_col. "
            "Укажи его явно через --target-col."
        )
    return best_col


def _validate_columns(df: pd.DataFrame, text_col: str, target_col: str) -> None:
    missing = [c for c in (text_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Нет колонок в датасете: {missing}. Доступные: {list(df.columns)}")


def _handle_rare_classes(
    x: pd.Series,
    y: pd.Series,
    *,
    min_class_count: int,
    merge_to_other: bool,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    counts = y.value_counts(dropna=False)

    rare = counts[counts < min_class_count]
    if rare.empty:
        return x, y, {}

    rare_dict = {str(k): int(v) for k, v in rare.to_dict().items()}

    if merge_to_other:
        rare_classes = set(rare.index.astype(str).tolist())
        y2 = y.astype(str).apply(lambda v: "__OTHER__" if v in rare_classes else v)
        return x, y2, rare_dict

    # drop rare
    keep_classes = set(counts[counts >= min_class_count].index.astype(str).tolist())
    mask = y.astype(str).isin(keep_classes)
    return x[mask], y[mask], rare_dict


def train_and_save(cfg: TrainConfig) -> None:
    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"Не найден входной файл: {cfg.input_path}. "
            f"Ожидается путь вида data/raw/Обращения.xlsx"
        )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(cfg.input_path)
    if df.empty:
        raise ValueError("Excel-файл прочитан, но таблица пустая.")

    text_col = cfg.text_col or ("text" if "text" in df.columns else _infer_text_column(df))
    target_col = cfg.target_col or ("target" if "target" in df.columns else _infer_target_column(df, text_col))
    _validate_columns(df, text_col, target_col)

    LOGGER.info("Используем колонки: text_col=%s, target_col=%s", text_col, target_col)

    x = df[text_col].map(_clean_text)
    y = df[target_col].astype(str).fillna("").map(lambda s: s.strip())

    mask = (x.str.len() > 0) & (y.str.len() > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 30:
        raise ValueError(f"Слишком мало валидных строк после фильтрации: {len(x)}")

    x, y, rare_info = _handle_rare_classes(
        x,
        y,
        min_class_count=cfg.min_class_count,
        merge_to_other=cfg.merge_rare_to_other,
    )
    if rare_info:
        action = "объединены в __OTHER__" if cfg.merge_rare_to_other else "удалены"
        LOGGER.warning("Редкие классы (<%d) %s: %s", cfg.min_class_count, action, rare_info)

    if y.nunique() < 2:
        raise ValueError(
            "После обработки редких классов осталось <2 классов. "
            "Нужно больше данных или включить --merge-rare-to-other."
        )

    # split with stratify + fallback
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )
    except ValueError as e:
        LOGGER.warning("Stratify split не удался (%s). Делаю split без stratify.", e)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=None,
        )

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        ngram_range=(1, cfg.ngram_max),
    )

    clf = LogisticRegression(
        max_iter=cfg.max_iter,
        n_jobs=1,
        class_weight="balanced",
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)

    metrics = {
        "rows_total": int(len(df)),
        "rows_used": int(len(x)),
        "text_col": text_col,
        "target_col": target_col,
        "test_size": cfg.test_size,
        "random_state": cfg.random_state,
        "min_class_count": cfg.min_class_count,
        "merge_rare_to_other": cfg.merge_rare_to_other,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classes": sorted(y.unique().tolist()),
        "rare_classes": rare_info,
    }

    report = classification_report(y_test, y_pred, digits=4)

    (cfg.artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (cfg.artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    joblib.dump(vectorizer, cfg.artifacts_dir / "tfidf.joblib")
    joblib.dump(clf, cfg.artifacts_dir / "model.joblib")

    (cfg.artifacts_dir / "run_info.json").write_text(
        json.dumps(
            {
                "input_path": str(cfg.input_path),
                "artifacts_dir": str(cfg.artifacts_dir),
                "text_col": text_col,
                "target_col": target_col,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Готово. Артефакты: %s", cfg.artifacts_dir)
    LOGGER.info("accuracy=%.4f, f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train NLP classifier and save artifacts.")
    parser.add_argument("--input", default="data/raw/Обращения.xlsx", help="Path to Excel dataset.")
    parser.add_argument("--artifacts-dir", default="artifacts/nlp", help="Where to write artifacts.")
    parser.add_argument("--text-col", default=None, help="Text column name (optional).")
    parser.add_argument("--target-col", default=None, help="Target column name (optional).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=2000)

    parser.add_argument(
        "--min-class-count",
        type=int,
        default=2,
        help="Minimum samples per class to keep it (or merge to __OTHER__).",
    )
    parser.add_argument(
        "--merge-rare-to-other",
        action="store_true",
        help="Merge rare classes to __OTHER__ instead of dropping them.",
    )

    args = parser.parse_args()
    return TrainConfig(
        input_path=Path(args.input),
        artifacts_dir=Path(args.artifacts_dir),
        text_col=args.text_col,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
        max_iter=args.max_iter,
        min_class_count=args.min_class_count,
        merge_rare_to_other=args.merge_rare_to_other,
    )


def main() -> None:
    _setup_logging()
    cfg = _parse_args()
    train_and_save(cfg)


if __name__ == "__main__":
    main()
