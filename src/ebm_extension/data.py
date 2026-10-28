"""Native EBM schema and immutable historical partitions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.data import ROOT, FEATURES, NUMERIC, load_data, predictor_groups


def schema(frame):
    if len(frame.columns) != len(FEATURES) or set(frame.columns) != set(FEATURES):
        raise ValueError('Provide exactly the thirteen approved predictors.')
    out = frame[FEATURES].copy()
    for name in FEATURES:
        if name in NUMERIC:
            out[name] = pd.to_numeric(out[name], errors='raise').astype(float)
            if np.isinf(out[name]).any():
                raise ValueError('Infinite numeric input.')
        else:
            out[name] = out[name].map(lambda v: np.nan if pd.isna(v) else str(v))
    return out


def established_data():
    data = load_data()
    saved = pd.read_csv(ROOT/'results/accepted/splits.csv', dtype={'group': str})
    assert saved.row_id.tolist() == list(range(len(data)))
    groups = predictor_groups(data).astype(str).to_numpy()
    assert np.array_equal(groups, saved.group.to_numpy())
    dev = saved.loc[saved.partition == 'development', 'row_id'].to_numpy()
    test = saved.loc[saved.partition == 'test', 'row_id'].to_numpy()
    assert (len(dev), len(test)) == (7811, 1959)
    assert not set(groups[dev]) & set(groups[test])
    folds = []
    labels = saved.iloc[dev].cv_fold.to_numpy()
    for k in range(5):
        train, valid = np.flatnonzero(labels != k), np.flatnonzero(labels == k)
        assert not set(groups[dev[train]]) & set(groups[dev[valid]])
        folds.append((train, valid))
    return data, dev, test, folds, groups


def grouped_bags(groups, seed=42):
    groups = np.asarray(groups)
    bags = np.zeros((len(groups), 4), dtype=np.int8)
    for bag in range(4):
        train, valid = next(GroupShuffleSplit(n_splits=1, test_size=.15,
                                             random_state=seed+bag).split(groups, groups=groups))
        assert not set(groups[train]) & set(groups[valid])
        bags[train, bag], bags[valid, bag] = 1, -1
    return bags
