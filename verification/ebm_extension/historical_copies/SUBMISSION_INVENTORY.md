# Final submission file inventory — 23 October 2026

This is a proposed packaging allowlist, not an archive or a submission. All listed files exist. No accepted experiment artifact was changed.

The payload contains **124 listed files**, plus this inventory and `verification/submission_inventory.json`: **52,587,779 bytes (50.15 MiB)** before those two small index files and before archive compression. The largest file is `models/accepted/pipeline.joblib`: **21,494,573 bytes (20.50 MiB)**. Its identical completed-run RF copy is also retained. No selected individual file exceeds the illustrative 25 MiB review flag; this does not establish compliance with an unknown upload limit.

## Package these groups

| Group | Purpose |
|---|---|
| Executed notebook, README, verification, dependency files and entry points | Academic presentation, setup and reproduction |
| All six `src/` modules and `tests/test_assignment.py` | Complete training/inference implementation and checks |
| `.streamlit/config.toml` | Local dashboard configuration |
| `data/README.md` | Independent version-3 acquisition, checksum and attribution |
| `models/accepted/` and all of `results/accepted/` | Selected fitted model and accepted evidence |
| Complete `results/runs/main-local-20261020-v2/` | Original successful source run and all five model bundles |
| Listed direct `verification/` files | Historical execution, original progress, source provenance and dated review evidence |

The completed source run is required by the current overwrite-protection test, which verifies its completion manifest and every referenced file before refusing an overwrite. Do not omit that directory simply because exploratory runs are normally excluded. The selected RF pipeline appears both there and in the accepted deployment folder; this duplication supports the existing checks and has not been removed. The ignore rules retain this one completed run.

## Exclude these local development files

- `.venv/` — Installed environment; recreate from lockfile.
- `.pip-cache/` — Download/install cache.
- `.ipython/, .matplotlib/, .jupyter/` — Local runtime caches.
- `**/__pycache__/, **/*.pyc, notebook checkpoints` — Generated execution caches.
- `results/runs/main-local-20261020/` — Preliminary pilot with superseded group-key ordering.
- `verification/partial_promotion/` — Incomplete duplicate development copy.
- `data/car_price_dataset.csv, data/*.zip` — Acquire version 3 separately; redistribution rights unresolved.
- `.env*, .streamlit/secrets.toml, credential/key files` — Never include private credentials; none found.

These items were identified for exclusion, not deleted. The environment and pip cache alone contain roughly 1406.5 MiB and must not be uploaded as project evidence.

## Readiness findings

- No required implementation module or listed payload file is missing.
- All 81 lockfile distributions match the installed environment; pip check passed. A fresh install was not performed.
- 75 proposed text files were scanned for common private-key/token/literal-credential patterns; no suspected credentials were found. No secret/key filenames were found in the reviewed project scan. Binary models were checked by integrity hashes, not treated as text. This is not an exhaustive security guarantee.
- The notebook's code and stored execution evidence are preserved; 14 non-training tests passed with zero fit calls. No notebook cells or ML experiments were rerun.
- Dataset acquisition instructions are included, but the CSV download was not repeated. The existing local CSV checksum matches the accepted version.

Confirm the institutional upload/archive size limit before packaging. Confirm whether a separate report, cover sheet, student identifier, declaration or prescribed AI-use statement is mandatory; none can be inferred from the supplied rubric. If grading is offline, arrange legitimate access to the version-3 CSV. Raw dataset redistribution permission remains unresolved. Nothing was committed, pushed, published, deployed or submitted.

## Exact file list

SHA256 values are in `verification/submission_inventory.json`. The following list excludes only the two inventory documents themselves to avoid self-referential hashes and sizes; include both index files in the package.

| File | Bytes |
|---|---:|
| `.gitignore` | 372 |
| `.streamlit/config.toml` | 208 |
| `README.md` | 12,853 |
| `VERIFICATION.md` | 11,198 |
| `app.py` | 9,937 |
| `data/README.md` | 2,230 |
| `execute_notebook.py` | 1,322 |
| `models/accepted/fit.json` | 102 |
| `models/accepted/pipeline.joblib` | 21,494,573 |
| `requirements-lock.txt` | 1,494 |
| `requirements.txt` | 247 |
| `results/accepted/accepted_manifest.json` | 3,365 |
| `results/accepted/audit.json` | 1,294 |
| `results/accepted/catboost_predictions.csv` | 133,974 |
| `results/accepted/catboost_test.json` | 437 |
| `results/accepted/category_errors.json` | 1,398 |
| `results/accepted/complete.json` | 4,999 |
| `results/accepted/cv_folds.json` | 71,809 |
| `results/accepted/cv_results.json` | 17,153 |
| `results/accepted/development_cardinalities.csv` | 178 |
| `results/accepted/development_eda.json` | 873 |
| `results/accepted/development_summary.csv` | 891 |
| `results/accepted/figures/cv_comparison.png` | 26,701 |
| `results/accepted/figures/development_categories.png` | 78,891 |
| `results/accepted/figures/development_eda.png` | 159,426 |
| `results/accepted/figures/permutation_importance.png` | 44,336 |
| `results/accepted/figures/test_residuals.png` | 81,946 |
| `results/accepted/heldout_example.json` | 590 |
| `results/accepted/input_metadata.json` | 66,529 |
| `results/accepted/largest_errors.csv` | 1,996 |
| `results/accepted/median_predictions.csv` | 66,733 |
| `results/accepted/median_test.json` | 341 |
| `results/accepted/notebook_baselines.csv` | 458 |
| `results/accepted/permutation_importance.csv` | 690 |
| `results/accepted/pilot.json` | 2,297 |
| `results/accepted/price_band_errors.csv` | 282 |
| `results/accepted/protocol.json` | 6,442 |
| `results/accepted/random_forest_predictions.csv` | 134,727 |
| `results/accepted/random_forest_test.json` | 477 |
| `results/accepted/random_splits.csv` | 114,194 |
| `results/accepted/ridge_predictions.csv` | 134,007 |
| `results/accepted/ridge_test.json` | 360 |
| `results/accepted/selection.json` | 2,765 |
| `results/accepted/sensitivity.json` | 4,365 |
| `results/accepted/split_summary.json` | 195 |
| `results/accepted/splits.csv` | 381,845 |
| `results/accepted/test_results.json` | 2,323 |
| `results/accepted/xgboost_predictions.csv` | 116,883 |
| `results/accepted/xgboost_test.json` | 518 |
| `results/runs/main-local-20261020-v2/audit.json` | 1,294 |
| `results/runs/main-local-20261020-v2/catboost_predictions.csv` | 133,974 |
| `results/runs/main-local-20261020-v2/catboost_test.json` | 437 |
| `results/runs/main-local-20261020-v2/category_errors.json` | 1,398 |
| `results/runs/main-local-20261020-v2/complete.json` | 4,999 |
| `results/runs/main-local-20261020-v2/cv_folds.json` | 71,809 |
| `results/runs/main-local-20261020-v2/cv_results.json` | 17,153 |
| `results/runs/main-local-20261020-v2/development_cardinalities.csv` | 178 |
| `results/runs/main-local-20261020-v2/development_eda.json` | 873 |
| `results/runs/main-local-20261020-v2/development_summary.csv` | 891 |
| `results/runs/main-local-20261020-v2/figures/cv_comparison.png` | 26,701 |
| `results/runs/main-local-20261020-v2/figures/development_categories.png` | 78,891 |
| `results/runs/main-local-20261020-v2/figures/development_eda.png` | 159,426 |
| `results/runs/main-local-20261020-v2/figures/permutation_importance.png` | 44,336 |
| `results/runs/main-local-20261020-v2/figures/test_residuals.png` | 81,946 |
| `results/runs/main-local-20261020-v2/heldout_example.json` | 590 |
| `results/runs/main-local-20261020-v2/input_metadata.json` | 66,529 |
| `results/runs/main-local-20261020-v2/largest_errors.csv` | 1,996 |
| `results/runs/main-local-20261020-v2/median_predictions.csv` | 66,733 |
| `results/runs/main-local-20261020-v2/median_test.json` | 341 |
| `results/runs/main-local-20261020-v2/models/catboost/fit.json` | 106 |
| `results/runs/main-local-20261020-v2/models/catboost/model.cbm` | 3,318,484 |
| `results/runs/main-local-20261020-v2/models/catboost/pipeline.joblib` | 658,101 |
| `results/runs/main-local-20261020-v2/models/median/fit.json` | 108 |
| `results/runs/main-local-20261020-v2/models/median/pipeline.joblib` | 10,809 |
| `results/runs/main-local-20261020-v2/models/random_forest/fit.json` | 102 |
| `results/runs/main-local-20261020-v2/models/random_forest/pipeline.joblib` | 21,494,573 |
| `results/runs/main-local-20261020-v2/models/ridge/fit.json` | 108 |
| `results/runs/main-local-20261020-v2/models/ridge/pipeline.joblib` | 24,296 |
| `results/runs/main-local-20261020-v2/models/xgboost/fit.json` | 105 |
| `results/runs/main-local-20261020-v2/models/xgboost/model.ubj` | 975,433 |
| `results/runs/main-local-20261020-v2/models/xgboost/pipeline.joblib` | 267,026 |
| `results/runs/main-local-20261020-v2/models/xgboost/preprocessor.joblib` | 10,691 |
| `results/runs/main-local-20261020-v2/notebook_baselines.csv` | 458 |
| `results/runs/main-local-20261020-v2/permutation_importance.csv` | 690 |
| `results/runs/main-local-20261020-v2/pilot.json` | 2,297 |
| `results/runs/main-local-20261020-v2/price_band_errors.csv` | 282 |
| `results/runs/main-local-20261020-v2/protocol.json` | 6,442 |
| `results/runs/main-local-20261020-v2/random_forest_predictions.csv` | 134,727 |
| `results/runs/main-local-20261020-v2/random_forest_test.json` | 477 |
| `results/runs/main-local-20261020-v2/random_splits.csv` | 114,194 |
| `results/runs/main-local-20261020-v2/ridge_predictions.csv` | 134,007 |
| `results/runs/main-local-20261020-v2/ridge_test.json` | 360 |
| `results/runs/main-local-20261020-v2/selection.json` | 2,765 |
| `results/runs/main-local-20261020-v2/sensitivity.json` | 4,365 |
| `results/runs/main-local-20261020-v2/split_summary.json` | 195 |
| `results/runs/main-local-20261020-v2/splits.csv` | 381,845 |
| `results/runs/main-local-20261020-v2/test_results.json` | 2,323 |
| `results/runs/main-local-20261020-v2/xgboost_predictions.csv` | 116,883 |
| `results/runs/main-local-20261020-v2/xgboost_test.json` | 518 |
| `src/__init__.py` | 0 |
| `src/data.py` | 3,492 |
| `src/inference.py` | 2,393 |
| `src/models.py` | 3,976 |
| `src/reporting.py` | 4,700 |
| `src/training.py` | 19,427 |
| `tests/test_assignment.py` | 6,880 |
| `vehicle_price_assignment.ipynb` | 701,087 |
| `verification/activity_before.json` | 4,970 |
| `verification/browser_checks.json` | 1,307 |
| `verification/checks.json` | 667 |
| `verification/dashboard-desktop.jpg` | 36,045 |
| `verification/dashboard-mobile.jpg` | 33,244 |
| `verification/dashboard-overview.jpg` | 78,008 |
| `verification/notebook_progress_original.txt` | 9,637 |
| `verification/pip_check.txt` | 31 |
| `verification/presentation_before.json` | 27,191 |
| `verification/presentation_review.json` | 2,870 |
| `verification/presentation_tests.json` | 187 |
| `verification/presentation_tests.txt` | 1,988 |
| `verification/promotion_fix.json` | 521 |
| `verification/source_provenance.json` | 267 |
| `verification/training_before_promotion_fix.py` | 19,417 |
| `verification/unittest.txt` | 2,367 |
| `verify_project.py` | 2,618 |
