import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import boxcox

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSED_DIR = BASE_DIR / "Preprocessed"
PREPROCESSED_DIR.mkdir(exist_ok=True)


DATASET_DIRS = [BASE_DIR / "dataset", BASE_DIR / "Datasets", BASE_DIR / "datasets"]
DATASET_ALIASES = {
    "creditcard": ["creditcard", "CreditCard"],
    "7817_1_cleaned": ["7817_1_cleaned", "Amazon_Reviews"],
}


def _normalize(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def resolve_dataset_csv(dts: str) -> Path:
    candidates = [dts]
    candidates.extend(DATASET_ALIASES.get(dts, []))

    wanted = {_normalize(c) for c in candidates}
    direct = []
    for d in DATASET_DIRS:
        for c in candidates:
            direct.append(d / f"{c}.csv")

    for p in direct:
        if p.exists():
            return p

    for d in DATASET_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.csv"):
            if _normalize(p.stem) in wanted:
                return p

    searched = ", ".join(str(p) for p in direct)
    raise FileNotFoundError(f"Dataset CSV not found for '{dts}'. Looked in: {searched}")


def safe_boxcox(column: np.ndarray) -> np.ndarray:
    if column.size == 0:
        return column
    if np.all(column == column[0]):
        return column

    adjusted = np.where(column <= 0, 1e-6, column)
    try:
        transformed, _ = boxcox(adjusted)
        return transformed
    except Exception:
        return column


def _encode_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        encoded, _ = pd.factorize(series.fillna("NA").astype(str), sort=True)
        return pd.Series(encoded, index=series.index, dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return numeric.fillna(numeric.median() if not np.isnan(numeric.median()) else 0.0).astype(float)


def _split_features_labels(df: pd.DataFrame, dts: str):
    if dts == "7817_1_cleaned" and "reviews rating" in df.columns:
        drop_cols = [
            "id", "asins", "keys", "ean", "upc", "reviews doRecommend"
        ]
        existing_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop)
        y = _encode_series(df["reviews rating"])
        X = df.drop(columns=["reviews rating"])
    else:
        y = _encode_series(df.iloc[:, -1])
        X = df.iloc[:, :-1]

    X = X.apply(_encode_series)
    return X.to_numpy(dtype=float), y.to_numpy(dtype=int)


def preprocessing(dts: str):
    dataset_path = resolve_dataset_csv(dts)
    df = pd.read_csv(dataset_path)
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Dataset '{dts}' is empty or malformed: {dataset_path}")

    features, labels = _split_features_labels(df, dts)

    transformed = np.vstack([safe_boxcox(features[:, i]) for i in range(features.shape[1])]).T

    np.savetxt(PREPROCESSED_DIR / f"{dts}.csv", transformed, delimiter=",", fmt="%s")
    np.savetxt(PREPROCESSED_DIR / f"{dts}_label.csv", labels, delimiter=",", fmt="%s")
    # Compatibility with old naming in some code paths
    np.savetxt(PREPROCESSED_DIR / f"Preprocessed_{dts}.csv", transformed, delimiter=",", fmt="%s")


def transformation(dts):
    print("\nBox-Cox transformation of data..")
    preprocessing(dts)
