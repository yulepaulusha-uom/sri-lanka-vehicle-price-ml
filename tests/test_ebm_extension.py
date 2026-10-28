"""Scientific and inference checks for the separate EBM extension."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.data import ROOT, FEATURES, sha256
from src.ebm_extension.data import established_data, grouped_bags, schema
from src.ebm_extension.experiment import OUT, MODEL, CONFIGS
from src.ebm_extension.inference import explain_one
from src.deployment_runtime import load_ebm

class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.dev,cls.test,cls.folds,cls.groups=established_data()
        cls.model,cls.manifest=load_ebm()
        cls.values=cls.data.loc[11,FEATURES].to_dict()

    def test_exact_saved_partitions(self):
        seen=[]
        for a,b in self.folds:
            self.assertFalse(set(self.groups[self.dev[a]]) & set(self.groups[self.dev[b]]))
            seen.extend(self.dev[b])
        self.assertEqual(sorted(seen),sorted(self.dev))
        self.assertFalse(set(self.groups[self.dev])&set(self.groups[self.test]))

    def test_all_recorded_internal_bags(self):
        paths=list(OUT.glob('*_bags.npz'))
        self.assertEqual(len(paths),31)
        for path in paths:
            record=np.load(path)
            rows,bags=record['row_ids'],record['bags']
            self.assertTrue(set(rows)<=set(self.dev))
            np.testing.assert_array_equal(bags,grouped_bags(self.groups[rows]))
            for k in range(4):
                self.assertFalse(set(self.groups[rows[bags[:,k]>0]])&set(self.groups[rows[bags[:,k]<0]]))
            if 'fold' in path.stem:
                k=int(path.stem.split('fold')[1][0])
                expected=self.dev[self.folds[k][0]]
                np.testing.assert_array_equal(rows,expected)

    def test_schema_order_missing_and_exclusions(self):
        frame=pd.DataFrame([self.values])
        self.assertEqual(schema(frame[frame.columns[::-1]]).columns.tolist(),FEATURES)
        frame['Model']=None
        self.assertTrue(schema(frame).Model.isna().all())
        with self.assertRaises(ValueError):schema(frame.assign(Price=1))
        with self.assertRaises(ValueError):schema(frame.assign(**{'Engine (cc)':np.inf}))

    def test_unseen_missing_and_invalid(self):
        for replacement in ['__NEW_CATEGORY__',None]:
            result=explain_one(self.model,dict(self.values,Model=replacement))
            self.assertTrue(np.isfinite(result['prediction']))
            self.assertTrue(result['warnings'])
        with self.assertRaises(ValueError):explain_one(self.model,dict(self.values,YOM=2014.5))
        with self.assertRaises(ValueError):explain_one(self.model,dict(self.values,**{'Engine (cc)':float('inf')}))

    def test_reload_prediction_reproducibility(self):
        again=joblib.load(MODEL/'model.joblib')
        sample=schema(self.data.loc[self.test[:25],FEATURES])
        np.testing.assert_array_equal(self.model.predict(sample),again.predict(sample))
        np.testing.assert_array_equal(self.model.predict(sample),self.model.predict(sample))

    def test_exact_local_explanation(self):
        expected=json.loads((OUT/'local_example.json').read_text())
        result=explain_one(self.model,self.values)
        self.assertAlmostEqual(result['prediction'],expected['prediction'],places=9)
        self.assertEqual(len(result['terms']),13)
        self.assertLess(result['sum_difference'],1e-8)
        self.assertTrue(any(t['value']>0 for t in result['terms']))
        self.assertTrue(any(t['value']<0 for t in result['terms']))

    def test_negative_raw_prediction_not_clipped(self):
        with patch.object(self.model,'predict',return_value=np.array([-5.])),patch.object(self.model,'eval_terms',return_value=np.zeros((1,13))),patch.object(self.model,'intercept_',-5.):
            result=explain_one(self.model,self.values)
        self.assertEqual(result['prediction'],-5.)
        self.assertTrue(any('Negative' in s for s in result['warnings']))

    def test_native_global_explanation(self):
        global_data=self.model.explain_global().data()
        self.assertEqual(len(global_data['names']),13)
        self.assertTrue(np.isfinite(global_data['scores']).all())
        self.assertEqual(len(self.model.term_features_),13)
        self.assertTrue(all(len(t)==1 for t in self.model.term_features_))

    def test_budget_and_selection(self):
        cv=pd.read_csv(OUT/'cv_folds.csv')
        self.assertEqual(len(CONFIGS),6)
        self.assertEqual(len(cv),30)
        selection=json.loads((OUT/'selection.json').read_text())
        expected=cv.groupby('config').mae.mean().idxmin()
        self.assertEqual(selection['config']['id'],expected)
        self.assertTrue(selection['frozen_before_final_fit'])

    def test_protected_integrity(self):
        before=json.loads((ROOT/'verification/ebm_extension/integrity_before.json').read_text())
        from verification.ebm_extension.check_integrity import check_assignment_integrity
        self.assertGreater(check_assignment_integrity()['immutable_checked'],100)

    def test_original_activity_integrity_when_available(self):
        activity=ROOT.parent/'sri-lanka-vehicle-price-ml'
        if not activity.exists():
            self.skipTest('Original activity unavailable on this machine; external integrity check not performed.')
        before=json.loads((ROOT/'verification/ebm_extension/integrity_before.json').read_text())
        for name,digest in before['original_activity'].items():
            self.assertEqual(sha256(activity/name),digest,name)

    def test_frontend_ebm_prediction(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn('EBM predicted listing price',[m.label for m in app.metric])
        self.assertIn('Predicted listing price',[m.label for m in app.metric])
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(sum(m.label=='EBM predicted listing price' for m in app.metric),2)


    def test_sealed_outputs_unchanged_during_reuse(self):
        from src.ebm_extension.experiment import tune,finalize
        from src.ebm_extension.explain import explanations
        complete=json.loads((OUT/'complete.json').read_text())
        tune();finalize();explanations()
        for name,digest in (complete['files']|complete['source']).items():
            self.assertEqual(sha256(ROOT/name),digest,name)

    def test_real_negative_prediction_is_visible_in_frontend(self):
        records=pd.read_csv(OUT/'predictions.csv')
        row=int(records.loc[records.prediction<0,'row_id'].iloc[0])
        values=self.data.loc[row,FEATURES].to_dict()
        expected=explain_one(self.model,values)['prediction']
        app=AppTest.from_string('from app import show_ebm\nshow_ebm('+repr(values)+')',default_timeout=60).run()
        self.assertFalse(app.exception)
        shown=next(m.value for m in app.metric if m.label=='EBM predicted listing price')
        self.assertEqual(shown,f'{expected:,.2f}')
        self.assertLess(expected,0)
        self.assertTrue(any('Negative raw prediction' in w.value for w in app.warning))

    def test_fresh_run_paths_do_not_replace_reviewed_pointer(self):
        import subprocess,sys,os
        pointer=ROOT/'results/ebm_extension/manifest.json'
        before=sha256(pointer)
        env=dict(os.environ,EBM_RUN_ID='reproduction_path_check')
        result=subprocess.run([sys.executable,'-c',"from src.ebm_extension.experiment import OUT,MODEL; print(OUT.name,MODEL.name)"],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        self.assertIn('reproduction_path_check reproduction_path_check',result.stdout)
        self.assertEqual(sha256(pointer),before)
        self.assertFalse((ROOT/'results/ebm_extension/runs/reproduction_path_check').exists())

if __name__=='__main__': unittest.main()
