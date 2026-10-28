"""Integrity-checked EBM loading and exact additive single-record explanations."""
import json
import math
import numbers
import joblib
import numpy as np
import pandas as pd
from src.data import ROOT, FEATURES, NUMERIC, sha256
from .data import schema


def load_ebm():
    manifest=json.loads((ROOT/'results/ebm_extension/manifest.json').read_text())
    path=ROOT/manifest['model']
    if sha256(path)!=manifest['sha256']: raise ValueError('EBM model checksum differs.')
    return joblib.load(path),manifest


def explain_one(model, values):
    for col in NUMERIC:
        value=values.get(col)
        if value is not None and not pd.isna(value):
            if isinstance(value,bool) or not isinstance(value,numbers.Real):
                raise ValueError(col+': enter a numeric value.')
            if col=='Engine (cc)' and value<=0:
                raise ValueError('Engine capacity must be positive.')
    frame=schema(pd.DataFrame([values]))
    for col in NUMERIC:
        if pd.notna(frame[col].iloc[0]) and not math.isfinite(frame[col].iloc[0]):
            raise ValueError('Numeric inputs must be finite or missing.')
    if pd.notna(frame.YOM.iloc[0]) and int(frame.YOM.iloc[0])!=frame.YOM.iloc[0]:
        raise ValueError('Manufacturing year must be an integer.')
    terms=model.eval_terms(frame)[0]
    prediction=float(model.predict(frame)[0])
    intercept=float(np.asarray(model.intercept_).reshape(-1)[0])
    total=intercept+float(terms.sum())
    if not math.isfinite(prediction): raise ValueError('Nonfinite EBM prediction.')
    if model.link_!='identity' or not np.isclose(total,prediction,rtol=0,atol=1e-8):
        raise ValueError('Additive explanation does not match prediction.')
    warnings=[]
    for i,col in enumerate(FEATURES):
        value=frame[col].iloc[0]
        if pd.isna(value): warnings.append(col+': missing; learned missing bin may have no training support.')
        elif col not in NUMERIC and value not in model.bins_[i][0]:
            warnings.append(col+': unseen category; unknown bin contribution (normally zero with no support).')
        elif col in NUMERIC and not model.feature_bounds_[i][0]<=value<=model.feature_bounds_[i][1]:
            warnings.append(col+': outside development range; boundary-bin extrapolation is unreliable.')
    if prediction<0: warnings.append('Negative raw prediction is not a plausible price; it has not been clipped.')
    return dict(prediction=prediction,intercept=intercept,sum=total,sum_difference=abs(total-prediction),
                terms=[dict(term=name,value=float(v)) for name,v in zip(model.term_names_,terms)], warnings=warnings)
