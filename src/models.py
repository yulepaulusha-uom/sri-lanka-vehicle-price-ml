"""Small, explicit candidate spaces and leakage-safe model pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.data import FEATURES, NUMERIC, CATEGORICAL

NAMES = ["Median", "Ridge", "Random Forest", "XGBoost", "CatBoost"]

class CatFrame(BaseEstimator, TransformerMixin):
    """Preserve category names and fit numeric medians inside each training fold."""
    def fit(self, X, y=None):
        self.medians_ = X[NUMERIC].median()
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self
    def transform(self, X):
        out = X[FEATURES].copy()
        out[NUMERIC] = out[NUMERIC].fillna(self.medians_)
        for name in CATEGORICAL:
            out[name] = out[name].fillna("__MISSING__").astype(str)
        return out

REFERENCE_PARAMS = {
 "Median": {},
 "Ridge": {"alpha": 10.0},
 "Random Forest": {"n_estimators":150,"max_depth":18,"min_samples_split":4,"min_samples_leaf":2,"max_features":.8},
 "XGBoost": {"n_estimators":300,"learning_rate":.05,"max_depth":6,"subsample":.85,"colsample_bytree":.85,"reg_alpha":.1,"reg_lambda":2.},
 "CatBoost": {"iterations":600,"depth":6,"learning_rate":.05,"l2_leaf_reg":5.},
}

def candidate_parameters(name):
    if name == "Median": return [{}]
    if name == "Ridge": return [{"alpha": a} for a in [.1,1.,10.,100.,1000.]]
    if name == "Random Forest":
        space = dict(n_estimators=[150,300], max_depth=[12,18,None], min_samples_split=[4], min_samples_leaf=[2,4,8], max_features=[.6,.8])
        count = 8
    elif name == "XGBoost":
        space = dict(n_estimators=[300,600], learning_rate=[.03,.05,.1], max_depth=[3,5,6], subsample=[.7,.85], colsample_bytree=[.7,.85], reg_alpha=[.1,.5], reg_lambda=[2.,5.])
        count = 10
    else:
        space = dict(iterations=[400,600,800], depth=[4,6,8], learning_rate=[.03,.05,.1], l2_leaf_reg=[3.,5.,10.])
        count = 10
    candidates = [REFERENCE_PARAMS[name]]
    for params in ParameterSampler(space, n_iter=count*2, random_state=42):
        if params not in candidates: candidates.append(params)
        if len(candidates) == count: break
    return candidates

def make_model(name, params=None):
    params = REFERENCE_PARAMS[name] if params is None else params
    if name == "CatBoost":
        return Pipeline([("preprocessor", CatFrame()), ("regressor", CatBoostRegressor(
            loss_function="RMSE", cat_features=CATEGORICAL, random_seed=42, thread_count=2,
            allow_writing_files=False, verbose=False, boosting_type="Plain", **params))])
    numeric = [("imputer", SimpleImputer(strategy="median"))]
    if name == "Ridge": numeric.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(numeric), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))]), CATEGORICAL)
    ], sparse_threshold=1.0)
    if name == "Median": reg = DummyRegressor(strategy="median")
    elif name == "Ridge": reg = Ridge(solver="lsqr", tol=1e-5, max_iter=5000, **params)
    elif name == "Random Forest": reg = RandomForestRegressor(random_state=42, n_jobs=2, **params)
    else: reg = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=2, **params)
    return Pipeline([("preprocessor", prep), ("regressor", reg)])
