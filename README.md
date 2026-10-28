# Explainable Boosting Machine for Sri Lankan vehicle listing prices

**269112B · IT5514 Applied Machine Learning**

This assignment investigates an Explainable Boosting Machine (EBM) as a directly interpretable additive regressor. It is a **subsequent extension** of the completed five-model experiment. EBM was absent from the supplied eleven-lecture transcript, but its optimization uses gradient boosting; lecturer eligibility has not been formally confirmed. Random Forest remains the historical accuracy-selected winner.

## Report review and Azure preparation

The report preserves the verified findings, tables, XAI and equations. The published DOCX/PDF reflect the latest student-edited document. AI assistance is acknowledged in this README. No notebook, model, split or accepted result was changed during this preparation.

The app has passed clean **Linux CPython 3.12.16** verification with 45 exact runtime pins and no raw CSV. Use `deployment/requirements.txt` for runtime-only installation; use the research locks below for notebook reproduction. `src/deployment_runtime.py` handles the saved Windows manifest separators without editing the model or manifest. Deployment configuration, costs, OIDC setup and remaining approval steps are in [deployment/README.md](deployment/README.md).

The reviewed GitHub file list is in `deployment/github-files.json`. `python deployment/prepare_github.py` prepares a fresh `build/github/` review directory without initializing Git, committing or pushing. The Azure workflow is manual and defaults to verification only. The Azure payload is built separately with `python deployment/prepare.py`; it uses root `app.py` and runtime-only requirements. Do not deploy the whole research repository or a Windows virtual environment.

The approved GitHub destination is [sri-lanka-vehicle-price-ml](https://github.com/yulepaulusha-uom/sri-lanka-vehicle-price-ml). Azure region and budget remain pending; no Azure resources or public application deployment have been created. Azure runtime availability, names/quota and actual public proxy behavior must be rechecked after approval. The required 3–5-minute video and final submission ZIP remain uncreated.

## Review these deliverables

- `vehicle_price_ebm_extension.ipynb`: executed academic EBM notebook with actual training calls, tuning, evaluation, global/local XAI and discussion.
- `report/IT5514_Final_Assignment_269112B.docx` and `.pdf`: revised editable report and visually verified 10-page PDF (cover, eight main pages and references).
- `src/ebm_extension/`: native input schema, saved-partition validation, grouped internal bags, experiment, explanations and inference.
- `results/ebm_extension/runs/ebm-local-v1/`: sealed protocol, 30 fold checkpoints, selection, evaluation, predictions, support records, explanations and plots.
- `models/ebm_extension/ebm-local-v1/model.joblib`: the fitted EBM with learned continuous/category bins; `fit.json` records the final fit.
- `app.py`: existing Streamlit UI extended with EBM prediction and local explanation, retaining the historical RF reference.
- `VERIFICATION.md`, `verification/ebm_extension/` and `SUBMISSION_INVENTORY.md`: evidence, exact inventory and remaining concerns.

The original `vehicle_price_assignment.ipynb`, all five-model accepted artifacts, saved splits, old dependency files and original environment remain unchanged. The original activity folder is not needed for inference or reproduction. Prior report/UI/document versions are archived for provenance.

## Actual results

| Model | Study | Development CV MAE | Holdout MAE | RMSE | R² | Final fit s | Bundle MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Median | Historical accepted | 28.3609 | 27.8855 | 44.0346 | -0.0602 | 0.083 | 0.010 |
| Ridge | Historical accepted | 8.5247 | 8.4133 | 21.4865 | 0.7476 | 0.203 | 0.023 |
| Random Forest | Historical accepted | 5.6002 | 4.9093 | 19.0170 | 0.8023 | 52.429 | 20.499 |
| XGBoost | Historical accepted | 6.3770 | 5.9851 | 19.5078 | 0.7919 | 1.753 | 1.195 |
| CatBoost | Historical accepted | 6.8716 | 6.5747 | 19.9406 | 0.7826 | 28.656 | 3.792 |
| EBM | Subsequent extension | 10.3020 | 9.7644 | 23.9079 | 0.6875 | 7.870 | 0.141 |

EBM did not improve on Ridge or the historical tree ensembles. This is not a new joint six-model selection experiment. The same saved outer rows/folds are reused, but the holdout had already been inspected, and EBM adds group-preserving internal early stopping. Historical and extension results retain separate provenance. R² is not percentage accuracy. Timings and artifact sizes are approximate cost comparisons, not controlled hardware benchmarks.

C4 was selected by lowest mean grouped CV MAE: **10.3020 ± 0.7134** (sample fold SD), with max_bins 256, learning_rate 0.02 and min_samples_leaf 20. The other candidates and full fixed parameters are in `protocol.json`; at most six configurations and 30 outer fits were used. No historical algorithm was refitted. Final model bytes: **147,531**.

The EBM produced **38 negative raw holdout estimates (1.94%)**. They remain in metrics and trigger a warning rather than being silently clipped. Among 20 listings above 200 units, EBM MAE was 91.51. These are material limitations.

Year, Brand and engine capacity have the largest native EBM importance. Their effects are predictive associations, not causal changes in value. EBM importance (average absolute term contribution) differs from historical RF permutation importance (MAE increase after shuffling); the two cannot be treated as numerically equivalent measures.

The predeclared Audi A3 row has actual price 139.00, EBM prediction **149.404204**, intercept **53.441960** and all signed terms summing to **95.962244**. The additive identity differs from prediction by only **2.84e-14**. Its historical RF prediction remains **141.456916**.

## Dataset acquisition and attribution

Obtain [Prasad Nirmal’s SriLankan Vehicle(Car) Price Dataset, Kaggle version 3](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset). Follow `data/README.md` to place `car_price_dataset.csv` in `data/`. Required SHA256:

`4208d2673cf4d9c8d3d6042dc8a1b97004671812441975f5879550059fbce1ab`

The loader checks this exact version. Removing the exported index and 18 exact duplicate rows leaves 9,770 rows with no missing values in this file. The 13 predictors retain year and engine capacity as continuous and the other eleven as nominal. Mileage is omitted because every cleaned row satisfies `Millage(KM) = (2025 - YOM) * 11000`; it adds no independent odometer information. Date is excluded.

Recorded advertisements cover December 2024 to February 2025, from ikman.lk and Riyasewana.com. The target is historical advertised price in **unconfirmed original dataset units**, not current valuations or verified sale prices. The source's historical licence record does not independently resolve underlying marketplace redistribution rights. Raw CSV redistribution is excluded from the proposed package by default; retain attribution and check rights before sharing it. Acquisition instructions are independent of the original class activity. The existing local CSV was verified; a fresh Kaggle download was not performed during this extension.

## Setup and local dashboard

Use Python 3.12 and PowerShell in the assignment root. Create the isolated extension environment, leaving any existing `.venv` unchanged:

```powershell
py -3.12 -m venv .venv-ebm
.\.venv-ebm\Scripts\python.exe -m pip install -r requirements-ebm-lock.txt
.\.venv-ebm\Scripts\python.exe -m pip check
.\.venv-ebm\Scripts\python.exe -m streamlit run app.py
```

The exact extension lock contains the historical package versions plus `interpret-core==0.7.8`. `requirements-ebm.txt` documents that relationship. A fresh isolated installation and `pip check` were completed locally. No unrelated package upgrades were introduced.

Open [localhost:8502](http://127.0.0.1:8502). This is a local-only address, not a deployed service. The UI needs the saved models/results, but no CSV. **Predict a listing** returns the historical RF reference and new EBM estimate with signed waterfall and term table. **EBM explanations** shows separately recorded EBM metrics/global effects and the negative-output limitation. The held-out demonstration gives RF 141.46, EBM 149.40 and actual 139.00. Training-observed categories/ranges guide entry, rare categories trigger support warnings, and backend inference supports unknown/missing bins with warnings. Finite unknown predictions do not establish reliability. Only load trusted joblib artifacts with compatible versions. Git attributes preserve exact file bytes across platforms so accepted artifact checksums remain valid.

## Reproducing the EBM notebook

`src/ebm_extension/data.py` reads **the saved** `results/accepted/splits.csv`, verifying row IDs and group hashes. It reuses 7,811 development rows, 1,959 holdout rows and five saved GroupKFold assignments. It never uses a new outer split. Inside each training partition, four signed bags reserve 15% of groups using seeds 42–45; no group crosses training/internal validation or outer validation. Native bin fitting occurs within these training calls.

The notebook calls actual reusable training functions. Its first execution trained three folds, stopped at the 512 MiB available-RAM safeguard, and preserved them. After the user closed other apps, it resumed the remaining 27 folds with the same methodology. Actual completed training execution is archived in `verification/ebm_extension/notebook_training_execution.ipynb`; the earlier interruption is also retained. A later presentation verification executed the notebook with fitting forbidden and loaded the completed evidence. It does **not** claim another tuning search.

To inspect/re-execute the completed notebook without fitting:

```powershell
$env:EBM_VERIFY_ONLY='1'
.\.venv-ebm\Scripts\python.exe -m verification.ebm_extension.execute_notebook
Remove-Item Env:EBM_VERIFY_ONLY
```

For a deliberate **fresh reproduction**, acquire the checksum-matching CSV and use a new safe run name before starting the kernel. Clear verification-only mode:

```powershell
Remove-Item Env:EBM_VERIFY_ONLY -ErrorAction SilentlyContinue
$env:EBM_RUN_ID='ebm-reproduction-01'
.\.venv-ebm\Scripts\python.exe -m verification.ebm_extension.execute_notebook
Remove-Item Env:EBM_RUN_ID
```

This executes a development-only pilot, the same six configurations/30 CV fits, frozen selection, final fit, holdout scoring and explanations in separate model/result directories. The reproduced notebook is saved under that result run. The reviewed dashboard model is not silently replaced. Use the same new run ID to resume its interrupted folds. Do not reuse an ID for a changed protocol or source; the protocol mismatch check refuses changed settings. The sealed completed run is read-only during ordinary notebook reuse. Do not expand tuning or revise a model using this previously inspected holdout.

An interactive notebook kernel can be the `.venv-ebm` interpreter in VS Code; set the assignment root as working directory and the run ID before importing experiment modules. The original historical notebook uses its original environment and commands documented in the archived README. Reproducing its 170 fits is optional and was **not** done for this EBM extension.

## Resource findings and verification

- Python 3.12.10, interpret-core 0.7.8; native Windows runtime/API/bag support passed.
- Pilot: 2,000 development rows, 200-round cap, 0.9217 s, 156.06 MiB sampled peak process-tree RSS.
- Actual CV fit time: 149.47 s across 30 fits; peak sampled process RSS 204.93 MiB. Sampling is every 100 ms and can miss short peaks.
- Initial stop: available system RAM 504.77 MiB. Resumed fits: minimum 1,012.18 MiB. No safeguard was lowered. Searches are sequential, n_jobs 1, BLAS/OpenMP limits 2.
- Five of 120 CV bag counters and one final bag reached the configured cyclic cap. Native stopping values are retained; more tuning was not added.
- Model reload, prediction reproducibility, native global XAI, local contribution sums, exact outer/internal groups and legacy RF behavior passed.
- All 166 immutable assignment hashes and 49 original-activity hashes match. The original activity, accepted historical results/model/splits/notebook/dependencies and environment were not modified.

Run checks without fitting any model:

```powershell
.\.venv-ebm\Scripts\python.exe -m verification.ebm_extension.run_checks
```

This includes original regression checks except the original small Ridge diagnostic fit, which is intentionally excluded. The original test source is unchanged. Additional tests cover EBM inference, bag assignments and protected hashes. Original-activity integrity checks are optional/skipped if that separate folder is unavailable on another machine; absence is not an integrity pass. `VERIFICATION.md` contains dated outcomes and real browser evidence. A hash-verified submission copy also passed 32 checks with only the two unavailable original-activity checks skipped, using the installed compatible interpreter and locally verified CSV; no model was refitted.

## Limitations and submission

Exact attributes cannot establish vehicle identity or eliminate near-duplicates. The holdout was already inspected. The search is small and additive; it does not prove optimality or a universal accuracy/interpretability trade-off. Rare categories, expensive listings, negative outputs and unresolved price units prevent authoritative pricing claims. No calibrated interval, future-market validation, causal conclusion or demographic fairness claim is supported. Town/brand/equipment may proxy social and geographic differences; demographic labels are absent.

OpenAI Codex assisted implementation, local execution, verification and writing. The student must understand the work, review the eligibility interpretation and adapt disclosure to institutional rules. The official deliverables include a PDF, source and a 3–5-minute demonstration video. The video is intentionally **not created**; the student will record it after reviewing the UI/report. Confirm upload limits, cover/declaration details and data rights before packaging. No final ZIP, video, Azure deployment or Moodle upload has been made.

Method sources: [InterpretML EBM overview](https://interpret.ml/docs/ebm.html), [regressor API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html), [scikit-learn grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html). Historical algorithm references and execution provenance remain in the archived comparative report and original notebook.
