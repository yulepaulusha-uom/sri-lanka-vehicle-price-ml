"""Sequential, checkpointed experiments for a small local CPU machine."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "2"
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from src.data import ROOT, FEATURES, CATEGORICAL, CSV_SHA256, load_data, make_splits, random_split, predictor_groups, audit_data, sha256
from src.models import NAMES, REFERENCE_PARAMS, candidate_parameters, make_model

PRICE_BINS = [0,25,50,100,200,float("inf")]
PRICE_LABELS = ["0-25","25-50","50-100","100-200","200+"]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x,"item") else str(x), allow_nan=False), encoding="utf-8")
    temp.replace(path)

def metrics(y, prediction):
    return dict(MAE=float(mean_absolute_error(y,prediction)), RMSE=float(np.sqrt(mean_squared_error(y,prediction))), R2=float(r2_score(y,prediction)) if len(y)>1 else None)

class ResourceMonitor:
    """Sample process RSS and available system RAM every 50 ms."""
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.min_available = psutil.virtual_memory().available
        self.start = time.perf_counter()
        def sample():
            process = psutil.Process()
            while not self.stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self.min_available = min(self.min_available, psutil.virtual_memory().available)
                self.stop.wait(.05)
        self.worker = threading.Thread(target=sample,daemon=True)
        self.worker.start()
        return self
    def __exit__(self,*args):
        self.stop.set()
        self.worker.join()
        self.seconds = time.perf_counter() - self.start
    def record(self):
        return dict(seconds=self.seconds, peak_rss_mb=self.peak/2**20, min_available_mb=self.min_available/2**20)

def check_resources(record):
    if record["peak_rss_mb"] > 2560 or record["min_available_mb"] < 512:
        raise RuntimeError("Resource threshold reached. Completed work is saved. Review before changing the protocol.")

def environment():
    packages=["numpy","pandas","scipy","scikit-learn","xgboost","catboost","streamlit","joblib","psutil"]
    return dict(python=platform.python_version(), platform=platform.platform(), logical_cpus=psutil.cpu_count(),
                total_ram_gb=psutil.virtual_memory().total/2**30,
                versions={p:importlib.metadata.version(p) for p in packages})

def protocol_signature():
    sources={p.name:sha256(p) for p in [ROOT/"src/data.py",ROOT/"src/models.py",ROOT/"src/training.py"]}
    protocol=dict(csv_sha256=CSV_SHA256, seed=42, test_size=.2, cv="GroupKFold(5)", search_n_jobs=1, estimator_threads=2,
                  primary_metric="MAE", selection="within 1% of minimum CV MAE, then lowest mean fit time", target="raw Price",
                  candidates={n:candidate_parameters(n) for n in NAMES}, sensitivity="fixed reference params, grouped and random; no tuning",
                  source_hashes=sources, environment=environment())
    return protocol, hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()

def new_run(label="academic"):
    path=ROOT/"results/runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+label)
    path.mkdir(parents=True,exist_ok=False)
    return path

def prepare_run(run_dir):
    run_dir=Path(run_dir).resolve()
    if ROOT not in run_dir.parents or run_dir == ROOT/"results/accepted":
        raise ValueError("Runs must be separate directories inside this assignment.")
    run_dir.mkdir(parents=True,exist_ok=True)
    protocol,signature=protocol_signature()
    manifest=run_dir/"protocol.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("Resume refused: code, environment or protocol changed. Start a new run.")
    else: write_json(manifest,dict(signature=signature,**protocol))
    return run_dir

def audit_and_split(run_dir):
    run_dir=prepare_run(run_dir)
    data=load_data()
    dev,test,folds=make_splits(data)
    write_json(run_dir/"audit.json",audit_data(data))
    groups=predictor_groups(data)
    split=pd.DataFrame({"row_id":np.arange(len(data)),"group":groups,"partition":"test","cv_fold":-1})
    split.loc[dev,"partition"]="development"
    for k,(_,b) in enumerate(folds): split.loc[dev[b],"cv_fold"]=k
    split.to_csv(run_dir/"splits.csv",index=False)
    random_dev,random_test=random_split(data)
    pd.DataFrame({"row_id":np.arange(len(data)),"partition":np.where(np.isin(np.arange(len(data)),random_dev),"train","test")}).to_csv(run_dir/"random_splits.csv",index=False)
    write_json(run_dir/"split_summary.json",dict(development_rows=len(dev),test_rows=len(test),group_overlap=0,cv_folds=5,
               random_train_rows=len(random_dev),random_test_rows=len(random_test),
               random_test_predictor_overlap=int(groups.iloc[random_test].isin(set(groups.iloc[random_dev])).sum())))
    return data,dev,test,folds

def pilot(run_dir):
    run_dir=prepare_run(run_dir)
    if (run_dir/"pilot.json").exists(): return json.loads((run_dir/"pilot.json").read_text())
    data,dev,_,folds=audit_and_split(run_dir)
    a,b=folds[0]
    rng=np.random.default_rng(42)
    train=dev[np.sort(rng.choice(a,size=min(2000,len(a)),replace=False))]
    valid=dev[b[:500]]
    records=[]
    for name in NAMES:
        model=make_model(name)
        with ResourceMonitor() as monitor, threadpool_limits(limits=2):
            model.fit(data.iloc[train][FEATURES],data.Price.iloc[train])
        record=dict(model=name,training_rows=len(train),validation_rows=len(valid),**monitor.record(),**metrics(data.Price.iloc[valid],model.predict(data.iloc[valid][FEATURES])))
        records.append(record)
        print("PILOT",json.dumps(record),flush=True)
        del model
        gc.collect()
    result=dict(environment=environment(),records=records,memory_note="Sampled process RSS, not total system allocation; includes Python and libraries.",
                projected_cv_seconds=sum(r["seconds"]*len(candidate_parameters(r["model"]))*5*(6250/2000) for r in records),
                estimate_note="Rough linear row scaling from reference configurations; deeper trees can cost more.")
    write_json(run_dir/"pilot.json",result)
    for r in records: check_resources(r)
    return result

def tune(run_dir):
    run_dir=prepare_run(run_dir)
    if not (run_dir/"pilot.json").exists(): raise ValueError("Run pilot before full search.")
    for record in json.loads((run_dir/"pilot.json").read_text())["records"]: check_resources(record)
    data,dev,_,folds=audit_and_split(run_dir)
    checkpoint=run_dir/"cv_folds.json"
    records=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    completed={(r["model"],r["candidate"],r["fold"]) for r in records}
    for name in NAMES:
        for candidate,params in enumerate(candidate_parameters(name)):
            for fold,(a,b) in enumerate(folds):
                if (name,candidate,fold) in completed: continue
                model=make_model(name,params)
                with ResourceMonitor() as monitor, threadpool_limits(limits=2):
                    model.fit(data.iloc[dev[a]][FEATURES],data.Price.iloc[dev[a]])
                prediction=model.predict(data.iloc[dev[b]][FEATURES])
                record=dict(model=name,candidate=candidate,fold=fold,params=params,**metrics(data.Price.iloc[dev[b]],prediction),**monitor.record())
                records.append(record)
                write_json(checkpoint,records)
                print("CV",name,candidate,fold,"MAE",round(record["MAE"],4),"seconds",round(record["seconds"],2),flush=True)
                del model,prediction
                gc.collect()
                check_resources(record)
    rows=[]
    for name in NAMES:
        for candidate,params in enumerate(candidate_parameters(name)):
            subset=[r for r in records if r["model"]==name and r["candidate"]==candidate]
            if len(subset)!=5: raise AssertionError("Incomplete CV candidate")
            row=dict(model=name,candidate=candidate,params=params)
            for key in ["MAE","RMSE","R2","seconds"]:
                row[key+"_mean"]=float(np.mean([r[key] for r in subset]))
                row[key+"_std"]=float(np.std([r[key] for r in subset],ddof=1))
            rows.append(row)
    write_json(run_dir/"cv_results.json",rows)
    best=[min((r for r in rows if r["model"]==n),key=lambda r:r["MAE_mean"]) for n in NAMES]
    minimum=min(r["MAE_mean"] for r in best)
    eligible=[r for r in best if r["MAE_mean"] <= minimum*1.01]
    winner=min(eligible,key=lambda r:r["seconds_mean"])
    selection=dict(winner=winner["model"],best_by_model=best,eligible_models=[r["model"] for r in eligible],
                   rule="Within 1% of lowest mean grouped CV MAE, then lowest mean fold fit time",test_used_for_selection=False)
    # Freeze this decision before fitting or scoring any final-test models.
    selection_path=run_dir/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
        raise ValueError("Frozen selection differs; start a new run.")
    write_json(selection_path,selection)
    return selection

def export_model(model,directory):
    directory.mkdir(parents=True,exist_ok=False)
    joblib.dump(model,directory/"pipeline.joblib",compress=3)
    reg=model.named_steps["regressor"]
    if reg.__class__.__name__=="XGBRegressor":
        reg.save_model(directory/"model.ubj")
        joblib.dump(model.named_steps["preprocessor"],directory/"preprocessor.joblib",compress=3)
    elif reg.__class__.__name__=="CatBoostRegressor": reg.save_model(str(directory/"model.cbm"))
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())

def evaluate(run_dir):
    run_dir=prepare_run(run_dir)
    if (run_dir/"complete.json").exists(): return json.loads((run_dir/"test_results.json").read_text())
    selection=json.loads((run_dir/"selection.json").read_text())
    data,dev,test,_=audit_and_split(run_dir)
    test_records=[]
    for chosen in selection["best_by_model"]:
        name=chosen["model"]
        slug=name.lower().replace(" ","_")
        artifact=run_dir/"models"/slug
        result_path=run_dir/(slug+"_test.json")
        if result_path.exists():
            test_records.append(json.loads(result_path.read_text()))
            continue
        if artifact.exists():
            # A model exported before interruption is reused rather than overwritten.
            model=joblib.load(artifact/"pipeline.joblib")
            resource=json.loads((artifact/"fit.json").read_text())
            size=sum(p.stat().st_size for p in artifact.iterdir() if p.name!="fit.json")
        else:
            model=make_model(name,chosen["params"])
            with ResourceMonitor() as monitor, threadpool_limits(limits=2): model.fit(data.iloc[dev][FEATURES],data.Price.iloc[dev])
            resource=monitor.record()
            # Store the model and timing together, then rename the completed directory.
            staging=artifact.with_name(artifact.name+"_staging")
            if staging.exists(): raise RuntimeError("Incomplete export staging directory requires review: "+str(staging))
            size=export_model(model,staging)
            write_json(staging/"fit.json",resource)
            staging.rename(artifact)
        start=time.perf_counter()
        prediction=model.predict(data.iloc[test][FEATURES])
        predict_seconds=time.perf_counter()-start
        reloaded=joblib.load(artifact/"pipeline.joblib")
        np.testing.assert_allclose(reloaded.predict(data.iloc[test][FEATURES]),prediction,rtol=1e-10,atol=1e-8)
        pred=pd.DataFrame({"row_id":test,"actual":data.Price.iloc[test].to_numpy(),"prediction":prediction})
        pred["residual"]=pred.actual-pred.prediction
        pred["absolute_error"]=abs(pred.residual)
        pred.to_csv(run_dir/(slug+"_predictions.csv"),index=False)
        record=dict(model=name,params=chosen["params"],**metrics(data.Price.iloc[test],prediction),**resource,
                    prediction_seconds=predict_seconds,artifact_bytes=size,reload_verified=True)
        write_json(result_path,record)
        test_records.append(record)
        print("TEST",json.dumps(record),flush=True)
        del model,reloaded,prediction
        gc.collect()
        check_resources(resource)
    write_json(run_dir/"test_results.json",test_records)
    sensitivity(run_dir,data,dev,test)
    interpret(run_dir,data,dev,test,selection)
    from src.reporting import make_figures
    make_figures(run_dir)
    files={str(p.relative_to(run_dir)):sha256(p) for p in run_dir.rglob('*') if p.is_file() and p.name!="complete.json"}
    write_json(run_dir/"complete.json",dict(winner=selection["winner"],file_hashes=files,completed_utc=datetime.now(timezone.utc).isoformat()))
    return test_records

def sensitivity(run_dir,data,dev,test):
    checkpoint=run_dir/"sensitivity.json"
    records=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    groups=predictor_groups(data)
    for protocol,(a,b) in [("grouped_fixed",(dev,test)),("random_fixed",random_split(data))]:
        overlap=int(groups.iloc[b].isin(set(groups.iloc[a])).sum())
        for name in NAMES:
            if any(r["protocol"]==protocol and r["model"]==name for r in records): continue
            model=make_model(name,REFERENCE_PARAMS[name])
            with ResourceMonitor() as monitor, threadpool_limits(limits=2): model.fit(data.iloc[a][FEATURES],data.Price.iloc[a])
            record=dict(protocol=protocol,model=name,params=REFERENCE_PARAMS[name],test_rows=len(b),overlap_rows=overlap,
                        **metrics(data.Price.iloc[b],model.predict(data.iloc[b][FEATURES])),**monitor.record())
            records.append(record)
            write_json(checkpoint,records)
            print("SENSITIVITY",protocol,name,round(record["MAE"],4),flush=True)
            del model
            gc.collect()
            check_resources(record)

def interpret(run_dir,data,dev,test,selection):
    slug=selection["winner"].lower().replace(" ","_")
    model=joblib.load(run_dir/"models"/slug/"pipeline.joblib")
    # Interpretation follows frozen selection; it never changes the model.
    if not (run_dir/"permutation_importance.csv").exists():
        with threadpool_limits(limits=2):
            importance=permutation_importance(model,data.iloc[test][FEATURES],data.Price.iloc[test],scoring="neg_mean_absolute_error",n_repeats=5,random_state=42,n_jobs=1)
        pd.DataFrame({"feature":FEATURES,"MAE_increase_mean":importance.importances_mean,"MAE_increase_std":importance.importances_std}).sort_values("MAE_increase_mean",ascending=False).to_csv(run_dir/"permutation_importance.csv",index=False)
    predictions=pd.read_csv(run_dir/(slug+"_predictions.csv"))
    predictions["price_band"]=pd.cut(predictions.actual,PRICE_BINS,labels=PRICE_LABELS)
    bands=predictions.groupby("price_band",observed=True).agg(count=("actual","size"),MAE=("absolute_error","mean"),mean_residual=("residual","mean"))
    bands.to_csv(run_dir/"price_band_errors.csv")
    predictions.nlargest(10,"absolute_error").merge(data[FEATURES],left_on="row_id",right_index=True).to_csv(run_dir/"largest_errors.csv",index=False)
    subgroup=[]
    for column in ["Brand","Model","Town"]:
        counts=data.iloc[dev][column].value_counts()
        frequencies=data.iloc[test][column].map(counts).fillna(0).to_numpy()
        for label,mask in [("unseen",frequencies==0),("rare_1_to_4",(frequencies>0)&(frequencies<5)),("seen_5_plus",frequencies>=5)]:
            if mask.any(): subgroup.append(dict(feature=column,group=label,count=int(mask.sum()),**metrics(predictions.actual.to_numpy()[mask],predictions.prediction.to_numpy()[mask])))
    write_json(run_dir/"category_errors.json",subgroup)
    row=int(test[0])
    record=data.iloc[row][FEATURES].to_dict()
    prediction=float(model.predict(data.iloc[[row]][FEATURES])[0])
    write_json(run_dir/"heldout_example.json",dict(row_id=row,rule="First row returned by primary held-out split; fixed before inspecting errors",features=record,actual=float(data.Price.iloc[row]),prediction=prediction,absolute_error=abs(prediction-data.Price.iloc[row])))
    metadata=dict(features=FEATURES,categories={c:sorted(data.iloc[dev][c].unique().tolist()) for c in CATEGORICAL},
                  brand_models={brand:sorted(part.Model.unique().tolist()) for brand,part in data.iloc[dev].groupby("Brand")},
                  numeric_ranges={c:[float(data.iloc[dev][c].min()),float(data.iloc[dev][c].max())] for c in ["YOM","Engine (cc)"]},
                  winner=selection["winner"],csv_sha256=CSV_SHA256,training_rows=len(dev),test_rows=len(test))
    write_json(run_dir/"input_metadata.json",metadata)

def accept_run(run_dir):
    run_dir=Path(run_dir).resolve()
    complete=json.loads((run_dir/"complete.json").read_text())
    for name,expected in complete["file_hashes"].items():
        if sha256(run_dir/name)!=expected: raise ValueError("Run hash mismatch: "+name)
    result_destination=ROOT/"results/accepted"
    model_destination=ROOT/"models/accepted"
    if result_destination.exists() or model_destination.exists():
        raise FileExistsError("Accepted output exists. Choose a new version explicitly; it will not be overwritten.")
    result_destination.mkdir()
    for p in run_dir.iterdir():
        if p.is_file(): shutil.copy2(p,result_destination/p.name)
        elif p.name=="figures": shutil.copytree(p,result_destination/p.name)
    winner=complete["winner"].lower().replace(" ","_")
    shutil.copytree(run_dir/"models"/winner,model_destination)
    write_json(result_destination/"accepted_manifest.json",dict(source_run=str(run_dir.relative_to(ROOT)),winner=complete["winner"],
               models={p.name:sha256(p) for p in model_destination.iterdir() if p.is_file()},
               results={p.name:sha256(p) for p in result_destination.iterdir() if p.is_file()}))
    return result_destination

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","pilot","tune","evaluate","accept"])
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    action={"audit":audit_and_split,"pilot":pilot,"tune":tune,"evaluate":evaluate,"accept":accept_run}[args.stage]
    action(Path(args.run))

if __name__=="__main__": main()
