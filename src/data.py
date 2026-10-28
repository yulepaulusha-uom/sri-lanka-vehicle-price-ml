"""Dataset provenance, deterministic cleaning and split definitions."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV_SHA256 = "4208d2673cf4d9c8d3d6042dc8a1b97004671812441975f5879550059fbce1ab"
SOURCE = "https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset"
FEATURES = ["Brand", "Model", "YOM", "Engine (cc)", "Gear", "Fuel Type", "Town", "Leasing", "Condition", "AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]
NUMERIC = ["YOM", "Engine (cc)"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data(path=None):
    path = Path(path) if path else ROOT / "data/car_price_dataset.csv"
    if sha256(path) != CSV_SHA256:
        raise ValueError("CSV checksum differs from approved Kaggle version 3.")
    raw = pd.read_csv(path)
    if raw.shape != (9788, 17) or set(raw.columns) != set(FEATURES + ["Unnamed: 0", "Date", "Millage(KM)", "Price"]):
        raise ValueError("Unexpected dataset schema or shape.")
    clean = raw.drop(columns="Unnamed: 0").drop_duplicates().reset_index(drop=True)
    if len(clean) != 9770 or clean.isna().any().any():
        raise ValueError("Unexpected cleaning outcome; investigate before training.")
    if not np.isfinite(clean[NUMERIC + ["Price", "Millage(KM)"]].to_numpy()).all() or (clean.Price <= 0).any():
        raise ValueError("Invalid numeric data or target.")
    if not np.isclose(clean["Millage(KM)"], (2025 - clean.YOM) * 11000).all():
        raise ValueError("Mileage no longer matches the documented derived field.")
    return clean

def predictor_groups(frame):
    # Hash only features, never target, date or row position.
    return pd.util.hash_pandas_object(frame[FEATURES], index=False)

def make_splits(data):
    groups = predictor_groups(data)
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(data, groups=groups))
    folds = list(GroupKFold(n_splits=5).split(data.iloc[dev], groups=groups.iloc[dev]))
    if set(groups.iloc[dev]) & set(groups.iloc[test]):
        raise AssertionError("Group leakage in primary holdout.")
    for a, b in folds:
        if set(groups.iloc[dev[a]]) & set(groups.iloc[dev[b]]):
            raise AssertionError("Group leakage in CV.")
    return dev, test, folds

def random_split(data):
    return train_test_split(np.arange(len(data)), test_size=.2, random_state=42)

def audit_data(data):
    groups = predictor_groups(data)
    return dict(source=SOURCE, version=3, csv_sha256=CSV_SHA256, original_rows=9788,
                cleaned_rows=len(data), exact_duplicates_removed=18, missing=data.isna().sum().to_dict(),
                unique_counts=data.nunique().to_dict(), predictor_groups=groups.nunique(),
                duplicate_predictor_rows=int(data.duplicated(FEATURES).sum()),
                conflicting_price_groups=int((data.groupby(FEATURES, dropna=False).Price.nunique()>1).sum()),
                mileage_formula_matches=int(np.isclose(data["Millage(KM)"], (2025-data.YOM)*11000).sum()),
                dates=[str(data.Date.min()), str(data.Date.max())],
                units="Original dataset price units; scale unconfirmed", target="Advertised listing price")
