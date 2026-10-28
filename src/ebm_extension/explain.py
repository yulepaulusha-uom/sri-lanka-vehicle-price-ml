"""Actual native term explanations, development support and historical comparison."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT, FEATURES
from .data import established_data, schema
from .experiment import OUT, MODEL, save
from .inference import explain_one


def explanations():
    if (OUT/'complete.json').exists():
        return (pd.read_csv(OUT/'term_importance.csv').sort_values('importance',ascending=False),
                json.loads((OUT/'local_example.json').read_text()), pd.read_csv(OUT/'comparison.csv'))
    model=joblib.load(MODEL/'model.joblib')
    data,dev,test,_,_=established_data()
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    importance=pd.DataFrame(dict(term=model.term_names_,importance=model.term_importances()))
    importance.to_csv(OUT/'term_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    ordered=importance.sort_values('importance')
    ax.barh(ordered.term,ordered.importance,color='#0f766e')
    ax.set_xlabel('Mean absolute term contribution (dataset price units)')
    fig.tight_layout(); fig.savefig(figures/'term_importance.png',dpi=160); plt.close(fig)
    for name,slug in [('YOM','year_effect'),('Engine (cc)','engine_effect')]:
        i=FEATURES.index(name)
        cuts=np.asarray(model.bins_[i][0])
        low,high=model.feature_bounds_[i]
        edges=np.r_[low,cuts,high]
        scores=np.asarray(model.term_scores_[i])[1:-1]
        assert len(scores)==len(edges)-1
        fig,(ax,support)=plt.subplots(2,1,figsize=(8,5),sharex=True,gridspec_kw={'height_ratios':[3,1]})
        ax.stairs(scores,edges,color='#0f766e',baseline=None); ax.axhline(0,color='gray',linewidth=.7)
        ax.set_ylabel('Additive contribution'); ax.set_title(name+' effect; association, not causation')
        support.hist(data.iloc[dev][name],bins=35,color='#94a3b8'); support.set_ylabel('Dev count'); support.set_xlabel(name)
        fig.tight_layout(); fig.savefig(figures/(slug+'.png'),dpi=160); plt.close(fig)
        pd.DataFrame(dict(lower=edges[:-1],upper=edges[1:],contribution=scores)).to_csv(OUT/(slug+'.csv'),index=False)
    name='Brand'; i=FEATURES.index(name)
    counts=data.iloc[dev][name].value_counts()
    supported=counts[counts>=50].index
    effects=pd.DataFrame([dict(category=c,development_count=int(counts[c]),contribution=float(model.term_scores_[i][model.bins_[i][0][c]])) for c in supported]).sort_values('contribution')
    effects.to_csv(OUT/'brand_effect.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh([f'{r.category} (n={r.development_count})' for r in effects.itertuples()],effects.contribution,color='#0f766e')
    ax.axvline(0,color='gray',linewidth=.7); ax.set_xlabel('Brand term contribution (supported categories: n ≥ 50)')
    fig.tight_layout(); fig.savefig(figures/'brand_effect.png',dpi=160); plt.close(fig)
    assert 11 in test
    local=explain_one(model,data.loc[11,FEATURES].to_dict())
    local.update(row_id=11,actual=float(data.Price.iloc[11]),features=data.loc[11,FEATURES].to_dict(),rule='Historical demonstration row, predeclared in protocol before EBM error inspection.')
    save(OUT/'local_example.json',local)
    terms=pd.DataFrame(local['terms']).sort_values('value')
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(terms.term,terms.value,color=['#b45309' if v<0 else '#0f766e' for v in terms.value])
    ax.axvline(0,color='gray',linewidth=.7)
    ax.set_title(f"Intercept {local['intercept']:.3f} + terms = {local['prediction']:.3f}")
    ax.set_xlabel('Signed contribution (dataset price units)')
    fig.tight_layout(); fig.savefig(figures/'local_explanation.png',dpi=160); plt.close(fig)
    historical=pd.DataFrame(json.loads((ROOT/'results/accepted/test_results.json').read_text()))
    selection=json.loads((ROOT/'results/accepted/selection.json').read_text())
    cv={r['model']:r['MAE_mean'] for r in selection['best_by_model']}
    comparison=historical[['model','MAE','RMSE','R2','seconds','artifact_bytes']].copy()
    comparison['cv_mae']=comparison.model.map(cv); comparison['study']='Historical accepted'
    e=json.loads((OUT/'evaluation.json').read_text())
    comparison.loc[len(comparison)]=['EBM',e['mae'],e['rmse'],e['r2'],e['training']['seconds'],e['model_bytes'],e['selected']['selected_cv_mae'],'Subsequent extension']
    comparison.to_csv(OUT/'comparison.csv',index=False)
    predictions=pd.read_csv(OUT/'predictions.csv')
    predictions['band']=pd.cut(predictions.actual,[0,25,50,100,200,np.inf])
    bands=predictions.groupby('band',observed=True).apply(lambda f:pd.Series(dict(n=len(f),mae=np.mean(np.abs(f.residual)),mean_residual=f.residual.mean())),include_groups=False)
    bands.to_csv(OUT/'price_band_errors.csv')
    predictions.assign(abs_error=predictions.residual.abs()).nlargest(15,'abs_error').to_csv(OUT/'largest_errors.csv',index=False)
    save(OUT/'xai_verification.json',dict(terms=len(model.term_names_),global_names=model.explain_global().data()['names'],
         global_scores=model.explain_global().data()['scores'],local_sum_difference=local['sum_difference'],
         link=model.link_,categorical_support_threshold=50,importance_definition='Development-weighted mean absolute native term contribution; not RF permutation MAE increase.'))
    return importance.sort_values('importance',ascending=False),local,comparison
