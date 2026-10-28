"""Student dashboard backed by accepted experiment artifacts."""
import os
for variable in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[variable]="2"
import json
import logging
import pandas as pd
import streamlit as st
from src.data import ROOT, FEATURES, SOURCE
from src.inference import load_bundle, predict_one

st.set_page_config(page_title="Sri Lanka | Vehicle Listing Prices",page_icon="🚗",layout="centered")
st.markdown("""<style>
.block-container {max-width:1080px;padding-top:4.5rem;padding-bottom:2rem}
h1 {letter-spacing:-.04em;font-size:2.3rem !important}
[data-testid="stMetric"] {background:white;border:1px solid #dce5eb;border-radius:12px;padding:14px}
.eyebrow {color:#0f766e;font-weight:650;letter-spacing:.13em;font-size:.75rem}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def resources(): return load_bundle()

def read_result(name): return json.loads((ROOT/"results/accepted"/name).read_text())

def main():
    st.markdown('<div class="eyebrow">MSC MACHINE LEARNING · FINAL ASSIGNMENT</div>',unsafe_allow_html=True)
    st.title("Vehicle listing prices, explained")
    st.write("A local regression study of Sri Lankan used vehicle advertisements.")
    try: model,metadata,manifest=resources()
    except Exception:
        logging.exception("Accepted artifacts could not be loaded")
        st.error("Accepted model or results could not be loaded. Check setup and artifact integrity.")
        st.stop()
    selection=read_result("selection.json")
    test=pd.DataFrame(read_result("test_results.json"))
    cv=pd.DataFrame(selection["best_by_model"])
    chosen=test.loc[test.model==metadata['winner']].iloc[0]
    a,b,c=st.columns(3)
    a.metric("Clean listings",f"{metadata['training_rows']+metadata['test_rows']:,}")
    b.metric("Selected model",metadata['winner'])
    c.metric("Grouped test MAE",f"{chosen.MAE:.3f}")
    st.caption("Model selected using development cross-validation. Prices and errors use original dataset units.")
    overview,predict,results,limits=st.tabs(["Overview","Predict a listing","Model results","Limitations"])
    with overview:
        st.subheader("The study")
        st.write("Compare a median baseline, Ridge Regression, Random Forest, XGBoost and CatBoost using 13 vehicle and listing attributes.")
        st.write(f"The primary experiment reserves {metadata['test_rows']:,} records with no identical predictor combinations shared with training. Five-fold grouped cross-validation guides model selection.")
        st.markdown(f"Dataset: [Prasad Nirmal · Kaggle version 3]({SOURCE}). Listings cover December 2024 to February 2025.")
        st.info("This estimates historical advertised listing prices. It does not establish current market value or an expected sale price. The currency scale remains unconfirmed.")
        st.image(str(ROOT/"results/accepted/figures/development_eda.png"),caption="Development data only: prices, manufacturing year and engine capacity.")
    with predict:
        st.subheader("Vehicle details")
        st.caption("Choose categories observed in primary training data. Mileage is excluded because it is derived from manufacturing year.")
        brands=list(metadata["brand_models"])
        default=brands.index("SUZUKI") if "SUZUKI" in brands else 0
        brand=st.selectbox("Brand",brands,index=default,key="brand")
        values={"Brand":brand}
        with st.form("vehicle_form"):
            left,right=st.columns(2)
            with left:
                values["Model"]=st.selectbox("Model",metadata['brand_models'][brand],key="model_"+brand)
                low,high=metadata['numeric_ranges']['YOM']
                values['YOM']=st.number_input("Year of manufacture",min_value=int(low),max_value=int(high),value=min(max(2014,int(low)),int(high)),step=1)
                for col in ['Gear','Town','Condition']:
                    options=metadata['categories'][col]
                    preferred={'Town':'Colombo','Condition':'USED','Fuel Type':'Petrol'}.get(col)
                    values[col]=st.selectbox(col,options,index=options.index(preferred) if preferred in options else 0,key=col)
            with right:
                low,high=metadata['numeric_ranges']['Engine (cc)']
                values['Engine (cc)']=st.number_input("Engine capacity (cc)",min_value=low,max_value=high,value=min(max(1000.,low),high),step=1.)
                for col in ['Fuel Type','Leasing']:
                    options=metadata['categories'][col]
                    preferred={'Town':'Colombo','Condition':'USED','Fuel Type':'Petrol'}.get(col)
                    values[col]=st.selectbox(col,options,index=options.index(preferred) if preferred in options else 0,key=col)
            st.markdown("**Equipment**")
            left,right=st.columns(2)
            for i,col in enumerate(['AIR CONDITION','POWER STEERING','POWER MIRROR','POWER WINDOW']):
                with left if i%2==0 else right:
                    options=metadata['categories'][col]
                    values[col]=st.selectbox(col.replace('_',' ').title(),options,index=options.index('Available') if 'Available' in options else 0,key=col)
            submit=st.form_submit_button("Predict listing price",type="primary",width="stretch")
        if submit:
            st.session_state["submitted_values"]=dict(values)
        if "submitted_values" in st.session_state:
            try:
                submitted_values=st.session_state["submitted_values"]
                prediction=predict_one(model,submitted_values,metadata)
                st.metric("Predicted listing price",f"{prediction:,.2f}")
                st.caption("Last submitted form · dataset units · no calibrated prediction interval.")
                with st.expander("Review input details"):
                    st.dataframe(pd.DataFrame([submitted_values],columns=FEATURES).T.rename(columns={0:'Value'}).astype(str),width="stretch")
            except ValueError as exc: st.error(str(exc))
            except Exception:
                logging.exception("Prediction failed")
                st.error("Prediction failed. Check the selected inputs and model files.")
        with st.expander("Demonstrate the held-out observation",expanded=st.session_state.get("show_heldout",False)):
            st.write("This record was selected by position in the held-out split, before inspecting its errors. It can contain categories absent from training.")
            if st.button("Run held-out example"):
                st.session_state["show_heldout"]=True
            if st.session_state.get("show_heldout",False):
                example=read_result('heldout_example.json')
                prediction=predict_one(model,example['features'],metadata,allow_unknown=True)
                st.dataframe(pd.DataFrame([example['features']]).T.astype(str),width="stretch")
                a,b=st.columns(2)
                a.metric("Example prediction",f"{prediction:,.2f}")
                b.metric("Actual advertised price",f"{example['actual']:,.2f}")
                st.caption(f"Absolute error: {abs(prediction-example['actual']):.2f} dataset units. One observation cannot establish overall performance.")
    with results:
        st.subheader("Development model selection")
        st.dataframe(cv[['model','MAE_mean','MAE_std','RMSE_mean','R2_mean','seconds_mean']].rename(columns={'seconds_mean':'Mean fold fit (s)'}),hide_index=True,width="stretch")
        st.caption(selection['rule']+'. Fold standard deviations describe variability, not confidence intervals.')
        st.subheader("Primary grouped holdout")
        shown=test[['model','MAE','RMSE','R2','seconds','artifact_bytes']].copy()
        shown['artifact_MiB']=shown.pop('artifact_bytes')/2**20
        st.dataframe(shown.rename(columns={'seconds':'Final fit (s)'}),hide_index=True,width="stretch")
        st.write("MAE is average absolute error. RMSE gives larger errors more weight. R² compares with a constant test-mean prediction and is not an accuracy percentage.")
        st.image(str(ROOT/'results/accepted/figures/test_residuals.png'))
        st.image(str(ROOT/'results/accepted/figures/permutation_importance.png'))
        st.caption("Permutation importance measures loss of predictive performance after shuffling an input. Related attributes can share importance; this is not causal evidence.")
        with st.expander("Separate split sensitivity study"):
            sensitivity=pd.DataFrame(read_result('sensitivity.json'))
            st.dataframe(sensitivity[['protocol','model','MAE','RMSE','R2','overlap_rows']],hide_index=True,width='stretch')
            st.write("Both split designs use the same fixed reference settings in this table. This secondary experiment did not select the deployed model. Historical activity results use a different protocol and are not proof of algorithmic improvement.")
    with limits:
        st.subheader("Read the predictions in context")
        st.markdown("""
- Historical advertised prices are not verified transactions or current valuations.
- Currency scale is unconfirmed, so no rupee or lakh conversion is applied.
- Exact attribute groups are not verified vehicle identities; near-duplicates may remain.
- Rare categories and expensive vehicles can produce large errors.
- Mileage matches a year-derived formula throughout this dataset and is excluded.
- This dataset was previously explored in a class activity; it is not external validation.
- No calibrated prediction interval or future-market evaluation is provided.
""")
        st.markdown(f"[Dataset attribution]({SOURCE}) · Source file verified by SHA256.")
    st.divider()
    st.caption('Accepted experiment: '+manifest['source_run']+' · Local educational demonstration')

if __name__=='__main__': main()
