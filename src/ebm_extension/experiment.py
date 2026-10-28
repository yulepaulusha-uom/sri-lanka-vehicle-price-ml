"""Bounded sequential experiment with per-fold checkpoints and explicit bags."""
import os
import re
import inspect
import json
import platform
import threading
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data import ROOT, FEATURES, NUMERIC, sha256
from .data import established_data, schema, grouped_bags

RUN = os.environ.get('EBM_RUN_ID', 'ebm-local-v1')
if not re.fullmatch(r'[A-Za-z0-9_-]+', RUN):
    raise ValueError('EBM_RUN_ID must be a single safe directory name.')
OUT = ROOT/'results/ebm_extension/runs'/RUN
MODEL = ROOT/'models/ebm_extension'/RUN
BASE = dict(feature_names=FEATURES, feature_types=['continuous' if c in NUMERIC else 'nominal' for c in FEATURES],
            objective='rmse', interactions=0, max_leaves=3, max_rounds=2000,
            outer_bags=4, inner_bags=0, early_stopping_rounds=50, validation_size=.15,
            greedy_ratio=0, smoothing_rounds=100, n_jobs=1, random_state=42,
            min_cat_samples=10, cat_smooth=10)
CONFIGS = [dict(id=f'C{i}', max_bins=b, learning_rate=r, min_samples_leaf=l)
           for i, (b,r,l) in enumerate([(128,.04,10),(256,.04,10),(128,.02,10),
                                        (128,.04,20),(256,.02,20),(256,.04,20)])]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x)), encoding='utf8')
    tmp.replace(path)


def metrics(y, pred):
    return dict(mae=float(mean_absolute_error(y,pred)), rmse=float(np.sqrt(mean_squared_error(y,pred))), r2=float(r2_score(y,pred)))


class Resources:
    def __enter__(self):
        self.peak = 0
        self.minimum_available = psutil.virtual_memory().available
        self.stop = threading.Event()
        def sample():
            while not self.stop.is_set():
                processes = [psutil.Process()]+psutil.Process().children(recursive=True)
                self.peak = max(self.peak, sum(p.memory_info().rss for p in processes if p.is_running()))
                self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
                self.stop.wait(.1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.seconds = time.perf_counter()-self.start
        self.stop.set()
        self.thread.join()
    def record(self):
        return dict(seconds=self.seconds, peak_process_tree_rss_bytes=self.peak,
                    minimum_system_available_bytes=self.minimum_available, sample_interval_seconds=.1)


def fit_model(X, y, groups, config, rounds=None):
    params = BASE | {k:v for k,v in config.items() if k!='id'}
    if rounds is not None: params['max_rounds'] = rounds
    bags = grouped_bags(groups)
    model = ExplainableBoostingRegressor(**params)
    with Resources() as resource:
        model.fit(schema(X), y, bags=bags)
    # EBM exposes training bag weights. Check supplied bags were actually used.
    np.testing.assert_allclose(model.bag_weights_, (bags>0).sum(axis=0))
    evidence = resource.record() | dict(best_iteration=np.asarray(model.best_iteration_).tolist(),
             train_counts=(bags>0).sum(axis=0).tolist(), validation_counts=(bags<0).sum(axis=0).tolist(),
             bag_weights=np.asarray(model.bag_weights_).tolist(), group_overlap=0)
    return model, bags, evidence


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(run=RUN, base=BASE, configurations=CONFIGS, selection='minimum mean grouped CV MAE',
        max_outer_fits=30, split_sha256=sha256(ROOT/'results/accepted/splits.csv'),
        csv_sha256=sha256(ROOT/'data/car_price_dataset.csv'), local_example_row_id=11,
        local_rule='Reuse the historical heldout demonstration row 11, declared before EBM error inspection.',
        holdout_disclosure='Previously inspected historical holdout; subsequent extension, not independent untouched benchmark.',
        internal_validation='Four explicit group-preserving 15% group validation bags within each training partition; seeds42..45.',
        rationale='Small additive search varies bin resolution, shrinkage and leaf support; interactions excluded for direct explanation.')
    path = OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == protocol
    else: save(path, protocol)
    return protocol


def pilot():
    prepare()
    path = OUT/'pilot.json'
    if path.exists(): return json.loads(path.read_text())
    data, dev, _, _, groups = established_data()
    idx = dev[:2000]
    model, bags, resource = fit_model(data.loc[idx,FEATURES], data.Price.iloc[idx], groups[idx], CONFIGS[0], rounds=200)
    sample = schema(data.loc[idx[:2],FEATURES])
    pred = model.predict(sample)
    unknown = sample.copy(); unknown.loc[unknown.index[0],'Model']='__UNSEEN_EBM_TEST__'
    missing = sample.copy(); missing.loc[missing.index[0],'Model']=np.nan
    assert np.isfinite(model.predict(unknown)).all() and np.isfinite(model.predict(missing)).all()
    joblib.dump(model, OUT/'pilot.joblib')
    np.testing.assert_allclose(pred, joblib.load(OUT/'pilot.joblib').predict(sample), rtol=0,atol=1e-10)
    import importlib.metadata as md
    result = dict(rows=len(idx), rounds_cap=200, resources=resource,
        versions={p:md.version(p) for p in ['interpret-core','numpy','pandas','scikit-learn','joblib','psutil']},
        python=platform.python_version(), constructor=str(inspect.signature(ExplainableBoostingRegressor)),
        fit_signature=str(inspect.signature(model.fit)), unknown_prediction=model.predict(unknown).tolist(),
        missing_prediction=model.predict(missing).tolist(), export_reload_pass=True,
        estimate_full_search_seconds_upper_linear=resource['seconds']*(7811/2000)*10*30)
    save(path,result)
    return result


def tune():
    if (OUT/'complete.json').exists():
        return pd.read_csv(OUT/'cv_summary.csv')
    prepare()
    data, dev, _, folds, groups = established_data()
    all_rows = []
    for config in CONFIGS:
        for k,(a,b) in enumerate(folds):
            path = OUT/f'{config["id"]}_fold{k}.json'
            if path.exists(): record = json.loads(path.read_text())
            else:
                train, valid = dev[a],dev[b]
                model,bags,resource = fit_model(data.loc[train,FEATURES],data.Price.iloc[train],groups[train],config)
                record = dict(config=config['id'],fold=k,**metrics(data.Price.iloc[valid],model.predict(schema(data.loc[valid,FEATURES]))),resources=resource)
                np.savez_compressed(OUT/f'{config["id"]}_fold{k}_bags.npz',row_ids=train,bags=bags)
                save(path,record)
                if resource['peak_process_tree_rss_bytes']>2.5*1024**3 or resource['minimum_system_available_bytes']<512*1024**2:
                    raise RuntimeError('Resource safeguard reached; completed fold saved. Review before continuing.')
            all_rows.append(record)
        print(config['id'], 'five folds recorded')
    rows = pd.DataFrame([{k:v for k,v in r.items() if k!='resources'} | dict(seconds=r['resources']['seconds']) for r in all_rows])
    rows.to_csv(OUT/'cv_folds.csv',index=False)
    summary = rows.groupby('config').agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),r2_mean=('r2','mean'),seconds=('seconds','sum')).reset_index().sort_values(['mae_mean','config'])
    summary.to_csv(OUT/'cv_summary.csv',index=False)
    selection = dict(config=next(c for c in CONFIGS if c['id']==summary.iloc[0]['config']), criterion='minimum mean grouped CV MAE', outer_fits=30, selected_cv_mae=float(summary.iloc[0].mae_mean), frozen_before_final_fit=True)
    if (OUT/'selection.json').exists(): assert json.loads((OUT/'selection.json').read_text())==selection
    else: save(OUT/'selection.json',selection)
    return summary


def finalize():
    # A finished run is loaded rather than refitted or rescored.
    if (OUT/'evaluation.json').exists(): return json.loads((OUT/'evaluation.json').read_text())
    selection = json.loads((OUT/'selection.json').read_text())
    data,dev,test,_,groups = established_data()
    MODEL.mkdir(parents=True,exist_ok=True)
    path = MODEL/'model.joblib'
    if path.exists():
        model=joblib.load(path); resource=json.loads((MODEL/'fit.json').read_text())
    else:
        model,bags,resource=fit_model(data.loc[dev,FEATURES],data.Price.iloc[dev],groups[dev],selection['config'])
        joblib.dump(model,path)
        save(MODEL/'fit.json',resource)
        np.savez_compressed(OUT/'final_bags.npz',row_ids=dev,bags=bags)
    Xtest=schema(data.loc[test,FEATURES])
    pred=model.predict(Xtest)
    np.testing.assert_allclose(pred,joblib.load(path).predict(Xtest),rtol=0,atol=1e-10)
    pd.DataFrame(dict(row_id=test,actual=data.Price.iloc[test].to_numpy(),prediction=pred,residual=data.Price.iloc[test].to_numpy()-pred)).to_csv(OUT/'predictions.csv',index=False)
    result=dict(**metrics(data.Price.iloc[test],pred),training=resource,model_bytes=path.stat().st_size,
                model_sha256=sha256(path),selected=selection,negative_predictions=int((pred<0).sum()),
                holdout='Previously inspected holdout; evaluated after frozen EBM selection.', reload_max_difference=0.)
    save(OUT/'evaluation.json',result)
    manifest=dict(run=RUN,model=str(path.relative_to(ROOT)),sha256=sha256(path),results=str(OUT.relative_to(ROOT)))
    save(OUT/'manifest.json',manifest)
    # A fresh reproduction never silently replaces the reviewed dashboard model.
    pointer=ROOT/'results/ebm_extension/manifest.json'
    if not pointer.exists(): save(pointer,manifest)
    return result
