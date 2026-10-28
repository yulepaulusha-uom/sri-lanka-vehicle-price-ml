"""Dataset-free inference/UI verification, with model fitting forbidden."""
import argparse,json,os,sys,unittest
from pathlib import Path
from unittest.mock import patch
parser=argparse.ArgumentParser();parser.add_argument('--app-root',required=True);parser.add_argument('--cases',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.app_root).resolve();cases=json.loads(Path(args.cases).read_text());output=Path(args.output).resolve()
os.chdir(root);sys.path.insert(0,str(root))
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
import numpy as np
from src.inference import load_bundle,predict_one,make_vehicle
from src.deployment_runtime import load_ebm,portable_relative
from src.ebm_extension.inference import explain_one
from sklearn.ensemble import RandomForestRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from streamlit.testing.v1 import AppTest

class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rf,cls.metadata,_=load_bundle();cls.ebm,_=load_ebm()
    def test_models_and_saved_preprocessing(self):
        sample=cases['heldout'];rf=predict_one(self.rf,sample['features'],self.metadata,allow_unknown=True)
        ebm=explain_one(self.ebm,sample['features'])
        self.assertAlmostEqual(rf,sample['rf_prediction'],places=7)
        self.assertAlmostEqual(ebm['prediction'],sample['ebm_prediction'],places=7)
        self.assertLess(ebm['sum_difference'],1e-8)
    def test_no_dataset_or_training_dependencies(self):
        self.assertFalse((root/'data/car_price_dataset.csv').exists())
        self.assertNotIn('src.models',sys.modules)
        self.assertNotIn('xgboost',sys.modules);self.assertNotIn('catboost',sys.modules)
    def test_invalid_and_unknown_inputs(self):
        values=cases['heldout']['features']
        with self.assertRaises(ValueError):make_vehicle(dict(values,YOM=True),self.metadata)
        with self.assertRaises(ValueError):explain_one(self.ebm,dict(values,**{'Engine (cc)':-1}))
        unknown=explain_one(self.ebm,dict(values,Model='__UNKNOWN_RUNTIME_TEST__'))
        self.assertTrue(np.isfinite(unknown['prediction']));self.assertTrue(unknown['warnings'])
    def test_portable_manifest_paths(self):
        self.assertEqual(portable_relative('models\\ebm_extension\\ebm-local-v1\\model.joblib'),'models/ebm_extension/ebm-local-v1/model.joblib')
        for bad in ['C:\\private\\model.joblib','../model.joblib','/absolute/path']:
            with self.assertRaises(ValueError):portable_relative(bad)
    def test_form_metrics_and_visualizations(self):
        app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
        self.assertFalse(app.exception);self.assertFalse(app.error)
        app.selectbox(key='brand').select('TOYOTA').run()
        self.assertEqual(app.selectbox(key='model_TOYOTA').options,self.metadata['brand_models']['TOYOTA'])
        app.button[0].click().run()
        self.assertFalse(app.exception);self.assertFalse(app.error)
        labels={m.label:m.value for m in app.metric}
        self.assertIn('EBM predicted listing price',labels);self.assertIn('Predicted listing price',labels)
        expected=json.loads((root/'results/ebm_extension/runs/ebm-local-v1/evaluation.json').read_text())
        self.assertEqual(labels['EBM holdout MAE'],f"{expected['mae']:.3f}")
        self.assertGreater(len(app.get('imgs')),0) if app.get('imgs') else self.assertGreater(len(app.get('plotly_chart')),0)
        app.button[1].click().run();self.assertFalse(app.exception)
    def test_real_negative_output_warning(self):
        sample=cases['negative'];result=explain_one(self.ebm,sample['features'])
        self.assertLess(result['prediction'],0)
        self.assertAlmostEqual(result['prediction'],sample['ebm_prediction'],places=7)
        app=AppTest.from_string('from app import show_ebm\nshow_ebm('+repr(sample['features'])+')',default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label=='EBM predicted listing price'),f"{result['prediction']:,.2f}")
        self.assertTrue(any('Negative raw prediction' in w.value for w in app.warning))

with patch.object(RandomForestRegressor,'fit',side_effect=AssertionError('Fitting prohibited')) as rf_guard,patch.object(ExplainableBoostingRegressor,'fit',side_effect=AssertionError('Fitting prohibited')) as ebm_guard:
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeChecks))
    record=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),fit_calls=rf_guard.call_count+ebm_guard.call_count,python=sys.version,platform=sys.platform)
    output.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
    if not result.wasSuccessful() or record['fit_calls']:raise SystemExit(1)
