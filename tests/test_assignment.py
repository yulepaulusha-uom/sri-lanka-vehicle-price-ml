"""Meaningful protocol, inference, artifact and Streamlit integration checks."""
import json
import unittest
import hashlib
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.data import ROOT, FEATURES, NUMERIC, CATEGORICAL, load_data, make_splits, predictor_groups, random_split
from src.models import make_model, candidate_parameters, NAMES
from src.inference import load_bundle, make_vehicle, predict_one

class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_data()
        cls.dev,cls.test,cls.folds=make_splits(cls.data)
    def test_verified_cleaning_and_features(self):
        self.assertEqual(len(self.data),9770)
        self.assertEqual(len(FEATURES),13)
        self.assertFalse(set(FEATURES)&{'Price','Date','Millage(KM)','Unnamed: 0'})
    def test_group_keys_match_actual_tuples(self):
        keys=predictor_groups(self.data)
        self.assertEqual(keys.nunique(),self.data[FEATURES].drop_duplicates().shape[0])
        grouped=self.data[FEATURES].copy()
        grouped['key']=keys
        self.assertTrue((grouped.groupby(FEATURES,dropna=False)['key'].nunique()==1).all())
    def test_holdout_and_all_cv_folds_have_no_group_overlap(self):
        groups=predictor_groups(self.data)
        self.assertFalse(set(groups.iloc[self.dev])&set(groups.iloc[self.test]))
        validation=[]
        for a,b in self.folds:
            self.assertFalse(set(groups.iloc[self.dev[a]])&set(groups.iloc[self.dev[b]]))
            validation.extend(b.tolist())
        self.assertEqual(sorted(validation),list(range(len(self.dev))))
        self.assertEqual((len(self.dev),len(self.test)),(7811,1959))
    def test_random_sensitivity_reproduces_known_overlap(self):
        a,b=random_split(self.data)
        keys=predictor_groups(self.data)
        self.assertEqual(int(keys.iloc[b].isin(set(keys.iloc[a])).sum()),297)
    def test_fold_local_preprocessing_with_novel_category(self):
        train=self.data.iloc[self.dev[:30]][FEATURES].copy()
        train['Engine (cc)']=1000.
        train.loc[train.index[0],'Engine (cc)']=np.nan
        model=make_model('Ridge',{'alpha':10.})
        model.fit(train,self.data.Price.iloc[self.dev[:30]])
        unseen=train.iloc[[0]].copy()
        unseen['Model']='__UNKNOWN_TEST_CATEGORY__'
        unseen['Engine (cc)']=4000.
        median=model.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['imputer'].statistics_[1]
        self.assertEqual(median,1000.)
        self.assertTrue(np.isfinite(model.predict(unseen)).all())
    def test_search_is_bounded_and_reproducible(self):
        self.assertEqual([len(candidate_parameters(n)) for n in NAMES],[1,5,8,10,10])
        self.assertEqual(candidate_parameters('CatBoost'),candidate_parameters('CatBoost'))

class AcceptedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.metadata,cls.manifest=load_bundle()
        data=load_data()
        dev,_,_=make_splits(data)
        cls.values=data.iloc[dev[0]][FEATURES].to_dict()
    def test_accepted_selection_uses_cv_only(self):
        selection=json.loads((ROOT/'results/accepted/selection.json').read_text())
        best=selection['best_by_model']
        minimum=min(r['MAE_mean'] for r in best)
        eligible=[r for r in best if r['MAE_mean']<=minimum*1.01]
        expected=min(eligible,key=lambda r:r['seconds_mean'])['model']
        self.assertEqual(selection['winner'],expected)
        self.assertFalse(selection['test_used_for_selection'])
    def test_heldout_export_prediction_matches(self):
        example=json.loads((ROOT/'results/accepted/heldout_example.json').read_text())
        actual=predict_one(self.model,example['features'],self.metadata,allow_unknown=True)
        self.assertAlmostEqual(actual,example['prediction'],places=7)
    def test_invalid_numbers_and_schema_rejected(self):
        for col,value in [('YOM',True),('YOM',2014.5),('Engine (cc)',float('inf')),('Engine (cc)',0),('YOM','2014')]:
            with self.subTest(col=col,value=value),self.assertRaises(ValueError):
                make_vehicle(dict(self.values,**{col:value}),self.metadata)
        with self.assertRaises(ValueError):make_vehicle(dict(self.values,Price=10),self.metadata)
        missing=dict(self.values)
        missing.pop('Town')
        with self.assertRaises(ValueError):make_vehicle(missing,self.metadata)
    def test_unseen_and_wrong_brand_model_rejected(self):
        for col in CATEGORICAL:
            with self.subTest(column=col),self.assertRaises(ValueError):
                make_vehicle(dict(self.values,**{col:'__UNKNOWN__'}),self.metadata)
        wrong=next(b for b,models in self.metadata['brand_models'].items() if self.values['Model'] not in models)
        with self.assertRaises(ValueError):make_vehicle(dict(self.values,Brand=wrong),self.metadata)
    def test_model_handles_unseen_categories(self):
        values=dict(self.values,Model='__UNKNOWN__')
        self.assertTrue(np.isfinite(predict_one(self.model,values,self.metadata,allow_unknown=True)))
    def test_reload_matches_batch_and_single_predictions(self):
        frame=make_vehicle(self.values,self.metadata)
        repeated=pd.concat([frame,frame],ignore_index=True)
        np.testing.assert_allclose(self.model.predict(repeated),[predict_one(self.model,self.values,self.metadata)]*2)
    def test_accepted_outputs_cannot_be_overwritten(self):
        from src.training import accept_run
        source=ROOT/self.manifest['source_run']
        with self.assertRaises(FileExistsError):accept_run(source)
    def test_streamlit_submission_and_brand_filter(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        self.assertFalse(app.exception)
        app.selectbox(key='brand').select('TOYOTA').run()
        self.assertEqual(app.selectbox(key='model_TOYOTA').options,self.metadata['brand_models']['TOYOTA'])
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn('Predicted listing price',[m.label for m in app.metric])
    def test_streamlit_heldout_demo(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        app.button[0].click().run()
        submitted=next(m.value for m in app.metric if m.label=='Predicted listing price')
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertIn('Example prediction',[m.label for m in app.metric])
        self.assertEqual(next(m.value for m in app.metric if m.label=='Predicted listing price'),submitted)

if __name__=='__main__':unittest.main()
