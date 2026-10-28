"""Phase A checks that do not require or fit the final EBM model."""
import json
import unittest
from pathlib import Path
import joblib
import numpy as np
from src.data import ROOT, FEATURES, sha256
from src.ebm_extension.data import established_data, grouped_bags, schema
from src.ebm_extension.experiment import OUT
from src.ebm_extension.inference import explain_one

class PhaseATests(unittest.TestCase):
    def test_protected_artifacts(self):
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

    def test_saved_outer_and_internal_partitions(self):
        data,dev,test,folds,groups=established_data()
        for train,valid in folds:
            bags=grouped_bags(groups[dev[train]])
            for b in range(4):
                self.assertFalse(set(groups[dev[train]][bags[:,b]>0]) & set(groups[dev[train]][bags[:,b]<0]))
                self.assertFalse(set(groups[dev[train]][bags[:,b]!=0]) & set(groups[dev[valid]]))
        self.assertEqual((len(dev),len(test)),(7811,1959))
        self.assertEqual(len(FEATURES),13)

    def test_pilot_inference_unknown_missing_and_exact_sum(self):
        model=joblib.load(OUT/'pilot.joblib')
        data,dev,*_=established_data()
        values=data.loc[dev[0],FEATURES].to_dict()
        for value in [values['Model'],'__NEW_CATEGORY__',None]:
            result=explain_one(model,dict(values,Model=value))
            self.assertTrue(np.isfinite(result['prediction']))
            self.assertLess(result['sum_difference'],1e-8)
        for col,value in [('YOM',True),('YOM',2014.5),('Engine (cc)',0),('Engine (cc)',float('inf'))]:
            with self.assertRaises(ValueError):explain_one(model,dict(values,**{col:value}))

    def test_completed_cv_checkpoint_bags(self):
        data,dev,test,folds,groups=established_data()
        paths=list(OUT.glob('*_fold*_bags.npz'))
        self.assertGreaterEqual(len(paths),3)
        for path in paths:
            record=np.load(path)
            k=int(path.stem.split('fold')[1][0])
            np.testing.assert_array_equal(record['row_ids'],dev[folds[k][0]])
            np.testing.assert_array_equal(record['bags'],grouped_bags(groups[record['row_ids']]))

if __name__=='__main__':unittest.main()
