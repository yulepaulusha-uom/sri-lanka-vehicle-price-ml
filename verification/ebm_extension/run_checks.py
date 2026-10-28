"""Regression checks with every historical and EBM fit forbidden."""
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='2'
import json
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from src.data import ROOT

suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
def leaves(node):
    for test in node:
        if isinstance(test,unittest.TestSuite):yield from leaves(test)
        else:yield test
selected=unittest.TestSuite(t for t in leaves(suite) if 'test_fold_local_preprocessing_with_novel_category' not in t.id())
with ExitStack() as stack:
    guards=[stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('Refitting prohibited during verification'))) for cls in [DummyRegressor,Ridge,RandomForestRegressor,XGBRegressor,CatBoostRegressor,ExplainableBoostingRegressor]]
    result=unittest.TextTestRunner(verbosity=2).run(selected)
    calls=sum(g.call_count for g in guards)
record=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),fit_calls=calls,
 excluded='Original diagnostic Ridge fit test; historical source unchanged.')
(ROOT/'verification/ebm_extension/all_checks.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
if not result.wasSuccessful() or calls:raise SystemExit(1)
