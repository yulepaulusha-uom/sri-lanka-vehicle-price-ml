"""Figures from development data and recorded, frozen test predictions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data import FEATURES, load_data, make_splits

def development_eda(run_dir):
    run_dir=Path(run_dir)
    directory=run_dir/"figures"
    directory.mkdir(exist_ok=True)
    data=load_data()
    dev,_,_=make_splits(data)
    train=data.iloc[dev]
    train.describe(include="all").T.to_csv(run_dir/"development_summary.csv")
    train[FEATURES].nunique().rename("unique").to_csv(run_dir/"development_cardinalities.csv")
    summary=dict(price_quantiles=train.Price.quantile([0,.25,.5,.75,.9,.95,.99,1]).to_dict(),price_skew=float(train.Price.skew()),
                 rare_models_under_5=int((train.Model.value_counts()<5).sum()),
                 whitespace_counts={c:int((train[c]!=train[c].str.strip()).sum()) for c in train.select_dtypes('object') if c!='Date'},
                 repeated_fractional_prices=train.Price.value_counts().head(10).to_dict(),
                 decisions="No speculative category corrections, price trimming, log target, or inferred odometer values.")
    from src.training import write_json
    write_json(run_dir/"development_eda.json",summary)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].hist(train.Price,bins=60,color="#0f766e")
    axes[0,0].set(title="Development listing prices",xlabel="Price (dataset units)",ylabel="Count")
    axes[0,1].hist(np.log1p(train.Price),bins=40,color="#2563eb")
    axes[0,1].set(title="Log scale for inspection only",xlabel="log(1 + price)")
    axes[1,0].scatter(train.YOM,train.Price,s=7,alpha=.25,color="#0f766e")
    axes[1,0].set(xlabel="Manufacturing year",ylabel="Price (dataset units)")
    axes[1,1].scatter(train['Engine (cc)'],train.Price,s=7,alpha=.25,color="#2563eb")
    axes[1,1].set(xlabel="Engine capacity (cc)",ylabel="Price (dataset units)")
    fig.tight_layout()
    fig.savefig(directory/"development_eda.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    train.Brand.value_counts().head(12).sort_values().plot.barh(ax=axes[0],color="#0f766e")
    axes[0].set(title="Most frequent brands",xlabel="Development records")
    top=train.Brand.value_counts().head(8).index
    axes[1].boxplot([train.loc[train.Brand==b,"Price"] for b in top],tick_labels=top,showfliers=False)
    axes[1].tick_params(axis='x',rotation=60)
    axes[1].set(title="Price by common brand (outliers hidden in this view)",ylabel="Dataset units")
    fig.tight_layout()
    fig.savefig(directory/"development_categories.png",dpi=150)
    plt.close(fig)
    return summary

def make_figures(run_dir):
    run_dir=Path(run_dir)
    directory=run_dir/"figures"
    directory.mkdir(exist_ok=True)
    selection=json.loads((run_dir/"selection.json").read_text())
    name=selection['winner']
    pred=pd.read_csv(run_dir/(name.lower().replace(' ','_')+"_predictions.csv"))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(pred.actual,pred.prediction,s=10,alpha=.35,color="#0f766e")
    maximum=max(pred.actual.max(),pred.prediction.max())
    axes[0].plot([0,maximum],[0,maximum],"--",color="#475569")
    axes[0].set(xlabel="Actual listing price",ylabel="Predicted listing price",title=name+": grouped holdout")
    axes[1].scatter(pred.prediction,pred.residual,s=10,alpha=.35,color="#2563eb")
    axes[1].axhline(0,linestyle="--",color="#475569")
    axes[1].set(xlabel="Predicted price",ylabel="Actual - predicted",title="Residuals (dataset units)")
    fig.tight_layout()
    fig.savefig(directory/"test_residuals.png",dpi=150)
    plt.close(fig)
    importance=pd.read_csv(run_dir/"permutation_importance.csv").sort_values("MAE_increase_mean")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(importance.feature,importance.MAE_increase_mean,xerr=importance.MAE_increase_std,color="#0f766e")
    ax.set(xlabel="Increase in MAE after permutation",title="Selected model: original input features")
    fig.tight_layout()
    fig.savefig(directory/"permutation_importance.png",dpi=150)
    plt.close(fig)
    cv=pd.DataFrame(selection['best_by_model']).sort_values("MAE_mean")
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(cv.model,cv.MAE_mean,yerr=cv.MAE_std,color="#0f766e",capsize=4)
    ax.set(ylabel="Grouped CV MAE",title="Five folds: mean and standard deviation")
    fig.tight_layout()
    fig.savefig(directory/"cv_comparison.png",dpi=150)
    plt.close(fig)
