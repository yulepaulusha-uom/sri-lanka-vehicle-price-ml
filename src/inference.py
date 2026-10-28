"""Load the accepted fitted pipeline and validate single-record inputs."""
import json
import math
import numbers
import joblib
import pandas as pd
from src.data import ROOT, FEATURES, CATEGORICAL, NUMERIC, sha256

def load_bundle():
    results=ROOT/"results/accepted"
    manifest=json.loads((results/"accepted_manifest.json").read_text())
    for name,checksum in manifest["models"].items():
        if sha256(ROOT/"models/accepted"/name)!=checksum: raise ValueError("Model artifact integrity check failed: "+name)
    for name,checksum in manifest["results"].items():
        if sha256(results/name)!=checksum: raise ValueError("Result integrity check failed: "+name)
    model=joblib.load(ROOT/"models/accepted/pipeline.joblib")
    metadata=json.loads((results/"input_metadata.json").read_text())
    return model,metadata,manifest

def make_vehicle(values,metadata,allow_unknown=False):
    if set(values)!=set(FEATURES): raise ValueError("Provide exactly the 13 approved predictors.")
    for column in NUMERIC:
        value=values[column]
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value):
            raise ValueError(column+" must be a finite number.")
        low,high=metadata["numeric_ranges"][column]
        if not low<=value<=high: raise ValueError(f"{column} must be within the training range {low:g} to {high:g}.")
    if int(values["YOM"])!=values["YOM"]: raise ValueError("Manufacturing year must be a whole number.")
    for column in CATEGORICAL:
        if not isinstance(values[column],str) or not values[column].strip(): raise ValueError(column+" must be a category string.")
        if not allow_unknown and values[column] not in metadata["categories"][column]: raise ValueError(column+": choose a training-observed category.")
    if not allow_unknown and values["Model"] not in metadata["brand_models"].get(values["Brand"],[]):
        raise ValueError("Choose a model observed for this brand in training.")
    return pd.DataFrame([values],columns=FEATURES)

def predict_one(model,values,metadata,allow_unknown=False):
    frame=make_vehicle(values,metadata,allow_unknown)
    prediction=float(model.predict(frame)[0])
    if not math.isfinite(prediction) or prediction<0: raise ValueError("Model returned an unsupported price; inspect this input.")
    return prediction
