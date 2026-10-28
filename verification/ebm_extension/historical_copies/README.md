# Sri Lankan Used Vehicle Listing Price Prediction

MSc Machine Learning final assignment comparing a median baseline, Ridge, Random Forest, XGBoost and CatBoost under group-aware validation. It extends an earlier class activity through a separate implementation, bounded tuning and error analysis. The original activity is not required to train the assignment or run its dashboard.

## What to review

- `vehicle_price_assignment.ipynb`: executed academic experiment, tables, figures and discussion.
- `src/`: supporting data, model, training, reporting and inference implementations.
- `results/accepted/`: accepted protocol, split assignments, fold scores, predictions and figures.
- `models/accepted/`: selected Random Forest pipeline and fit record used by the dashboard.
- `results/runs/main-local-20261020-v2/`: completed source run, including all five fitted model bundles.
- `VERIFICATION.md`: execution evidence, technical provenance and the dated presentation review.
- `SUBMISSION_INVENTORY.md`: exact proposed submission contents, sizes, exclusions and unresolved submission requirements.

## Dataset and acquisition

[Prasad Nirmal's SriLankan Vehicle(Car) Price Dataset, Kaggle version 3](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset) contains 9,788 records and 17 columns. Excluding the exported index and removing 18 exact duplicates leaves 9,770 rows. The target is historical advertised listing price. Source coverage is recorded as 3 December 2024 to 5 February 2025, from ikman.lk and Riyasewana.com.

The raw CSV is excluded from the proposed submission by default. Follow `data/README.md` to obtain **version 3**, extract `car_price_dataset.csv` into `data/`, and verify SHA256:

`4208d2673cf4d9c8d3d6042dc8a1b97004671812441975f5879550059fbce1ab`

The loader rejects a mismatching file. Dataset acquisition does not depend on the earlier activity. The historical Kaggle licence record is MIT; current metadata and underlying marketplace redistribution rights have not been independently established. Retain attribution and check rights before including raw data.

Currency scale is unresolved. Prices and errors remain in original dataset units; no rupee/lakh conversion is made. Advertisements are not verified transaction prices or current market valuations.

## Setup and dashboard

Use Python 3.12 and PowerShell in the extracted assignment root. For a fresh environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [localhost:8502](http://127.0.0.1:8502) after starting the server. The dashboard loads the accepted fitted preprocessing/model and recorded results, verifies their manifest checksums, and needs no CSV. The link is a local development address, not a deployed service. Only load trusted joblib files in a compatible environment.

## Reproducing the academic experiment

The submitted notebook retains outputs from the completed experiment. Presentation edits on 23 October 2026 did **not** rerun its code. Original code, execution counts, tables and charts were preserved; verbose tuning/evaluation streams were archived in `verification/notebook_progress_original.txt`.

For an intentional fresh reproduction, obtain the checksum-matching CSV, open the notebook in VS Code, select this assignment's `.venv` kernel, set the assignment root as the working directory, clear `ASSIGNMENT_RESUME_RUN` if set, and Restart Kernel → Run All. This performs actual baseline training, all 170 CV fits, final refits, export, sensitivity and interpretation in a new run directory. Accepted outputs remain unchanged.

The equivalent non-interactive command is:

```powershell
Remove-Item Env:ASSIGNMENT_RESUME_RUN -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe execute_notebook.py
```

This command executes training; it is not needed merely to inspect the submitted outputs. `execute_notebook.py` selects the project interpreter and saves notebook outputs. A fresh reproduction can also be run in stages:

```powershell
.\.venv\Scripts\python.exe -m src.training audit --run results/runs/my-new-experiment
.\.venv\Scripts\python.exe -m src.training pilot --run results/runs/my-new-experiment
# Inspect pilot.json before starting tuning.
.\.venv\Scripts\python.exe -m src.training tune --run results/runs/my-new-experiment
.\.venv\Scripts\python.exe -m src.training evaluate --run results/runs/my-new-experiment
```

Under an unchanged source/environment signature, reuse an interrupted run path to resume completed folds. The historical accepted run predates a promotion-only source fix, so the current runner intentionally refuses to resume that old signature; its exact execution source and provenance are preserved in verification. Start a new run for normal reproduction. See `VERIFICATION.md` for details.

A reviewed new run can be promoted with `python -m src.training accept --run <run-directory>`. Existing accepted paths cause refusal rather than replacement. No promotion or training was performed during this presentation review.

## Experimental design

- Thirteen predictors: Brand, Model, YOM, Engine (cc), Gear, Fuel Type, Town, Leasing, Condition, AIR CONDITION, POWER STEERING, POWER MIRROR and POWER WINDOW.
- Mileage is excluded because every cleaned row matches `(2025 - YOM) * 11000`; this establishes redundancy, not target leakage by itself. Index and date are also excluded from predictors.
- Primary split: GroupShuffleSplit, seed 42, with 20% of exact predictor groups held out; 7,811 development and 1,959 test rows. Five-fold GroupKFold is applied only to development rows.
- Fold-local imputation, encoding and relevant scaling prevent preprocessing leakage. CatBoost uses native categorical inputs and CPU Plain boosting. Detailed EDA uses development rows.
- Search sizes are 1/5/8/10/10 configurations for Median/Ridge/RF/XGBoost/CatBoost: 34 configurations and 170 fold fits. Candidate spaces and seeds are explicit in `src/models.py` and the accepted protocol.
- MAE selects each family's best configuration. Among family winners within 1% of the lowest mean MAE, the predeclared rule prefers the lowest mean fold fit time. Test performance never changes the selection. Fold SD is descriptive variability, not a confidence interval.
- RMSE, R², fit/prediction time and export size are also reported. Raw Price remains the target; log-price EDA is descriptive. No target trimming, LightGBM, stacking or deep learning is included.
- The separate sensitivity study fits identical fixed reference settings under grouped and random splitting. Neither sensitivity result selects the deployed model. Different test populations prevent interpreting score differences as the isolated effect of overlap.

Historical activity RF MAE/RMSE/R² was 5.8880/21.1365/0.8119; XGBoost was 6.9775/21.3378/0.8083. These remain historical context. Cross-protocol score changes do not prove algorithmic improvement, and Colab timings are not laptop speed benchmarks.

## Measured primary results

The accepted selection is Random Forest, chosen from grouped development CV before test evaluation.

| Model | CV MAE mean ± SD | Test MAE | Test RMSE | Test R² | Final fit (s) | Export bundle (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Median | 28.3609 ± 1.3442 | 27.8855 | 44.0346 | -0.0602 | 0.083 | 0.010 |
| Ridge | 8.5247 ± 0.6607 | 8.4133 | 21.4865 | 0.7476 | 0.203 | 0.023 |
| Random Forest | 5.6002 ± 0.5238 | 4.9093 | 19.0170 | 0.8023 | 52.429 | 20.499 |
| XGBoost | 6.3770 ± 0.4978 | 5.9851 | 19.5078 | 0.7919 | 1.753 | 1.195 |
| CatBoost | 6.8716 ± 0.6028 | 6.5747 | 19.9406 | 0.7826 | 28.656 | 3.792 |

All values are measured locally. CV uses five grouped development folds; test uses the 1,959-row grouped holdout. Bundle size includes companion native exports where present, so it is submitted artifact size, not a library-independent tree-size measure. All five pipeline reload checks passed on every test observation.

The 170 CV fits consumed 25.37 minutes of summed fit time (excluding notebook setup, pilot, checkpoint overhead, final refits and sensitivity runs). Highest sampled CV process RSS was 302.1 MiB; lowest sampled available system RAM was 1239.1 MiB.

RF has the lowest grouped CV MAE. XGBoost has the lowest CV RMSE and much shorter fit time; the predeclared MAE rule remains decisive. CatBoost did not improve MAE in this bounded study. These results do not establish superiority over the earlier activity, whose split and historical models differ.

The 20 primary-test observations above 200 units have MAE 67.02. Overall error should not be interpreted as uniform accuracy across price ranges. The held-out Audi A3 example has actual price 139.00 and prediction 141.46; it was chosen by split position, not error.

Fixed-parameter grouped sensitivity MAE was lower than random sensitivity MAE, reflecting different test populations rather than proving overlap harmless. Random RF reproduced the historical rounded metrics. Newly fitted random XGBoost (MAE 6.9265, RMSE 21.3634, R² 0.8079) differs slightly from the historical model; the precise cause was not established. Historical scores are retained separately.

## Resource limits and verification

Training is local, with sequential searches and two estimator threads. The 2,000-row pilot records fit time, sampled process RSS and available RAM. Resource thresholds stop after checkpointing if sampled RSS exceeds 2.5 GiB or available RAM falls below 512 MiB. No threshold was reached in the completed experiment. Permutation importance, residuals and subgroup errors describe the selected model after selection; they are not used to revise it.

The complete test suite is:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This includes one small diagnostic Ridge fit, not a refit of an accepted model. The presentation review instead ran the other 14 tests under fit-call guards and did not execute that diagnostic. Historical and review checks are dated separately in `VERIFICATION.md`.

`verify_project.py` runs the complete suite and checks notebook/accepted artifact integrity. Its original-activity comparison is optional when that separate folder/snapshot is absent on another machine; absence is reported as unavailable, not as an integrity pass. Include the completed source run listed in the inventory because the overwrite-protection test verifies its file manifest. The ignore rules retain this completed run while excluding other exploratory runs; use the inventory as the packaging allowlist.

## Limitations and acknowledgement

Rare/expensive listings can have large errors, the price scale is uncertain, derived mileage is unavailable as independent odometer data, and exact feature groups cannot establish vehicle identity. Near-duplicates and prior exploratory exposure remain limitations. There is no independent external/future-market validation, significance claim or calibrated prediction interval. Permuting correlated attributes can create implausible combinations and does not establish causation.

OpenAI Codex assisted code, experiment execution, documentation, presentation review and verification. The student must review the work, explain the methods and adapt the acknowledgement to institutional rules. The detailed rubric, report format, cover sheet, upload limit and university AI-use policy were not supplied; compliance with those unknown requirements is not claimed. Nothing has been committed, pushed, deployed or submitted.

## Method references

- Hoerl, A. E. and Kennard, R. W. (1970). [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634). Technometrics, 12(1), 55–67. Supports the regularized linear baseline.
- Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324). Machine Learning, 45, 5–32. Supports the randomized-tree ensemble comparator.
- Chen, T. and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). Supports the gradient-boosting comparator; its published benchmark results are not evidence about this vehicle dataset.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. and Gulin, A. (2018). [CatBoost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). NeurIPS 31. Motivates native categorical processing. This implementation uses CPU Plain boosting and does not claim to reproduce the paper's complete experimental settings.
