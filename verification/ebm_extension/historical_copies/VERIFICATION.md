# Verification evidence

Experiment/runtime verification below was recorded on 21 October 2026 (Asia/Colombo). A separate presentation-only review on 23 October 2026 is documented at the end. No commit, push, deployment or submission was performed.

## Separation and integrity

The assignment is entirely in `C:/ML-Projects/sri-lanka-vehicle-price-assignment`. The original activity remains read-only. All 49 monitored activity files match their pre-implementation SHA256 checksums, including source, notebooks, models, dataset, documentation, Git config/HEAD/index and virtual-environment configuration. Its final Git status is clean; optional locks were disabled and no Git configuration was changed. The original notebook, training code and recovery scripts were not executed during implementation. Dependencies were installed only into the assignment's new environment.

## Executed checks

| Check | Actual outcome |
|---|---|
| Dependencies | pip check: no broken requirements |
| Automated tests | 15 passed, including Streamlit form, brand filtering and persistent held-out demo checks |
| Notebook | All 13 code cells executed; zero error outputs; nbformat validation passed |
| Source | Python AST syntax checks passed |
| Dataset | Version-3 checksum matched; 9,788 original / 9,770 cleaned rows |
| Grouped holdout | 7,811 development / 1,959 test; zero shared predictor groups |
| CV | Five grouped folds; zero group overlap in every fold |
| Search | All 170 planned fold fits completed sequentially |
| Selection | Random Forest selected using development CV only |
| Export | All five reloaded pipelines matched every held-out prediction |
| Accepted artifacts | Model/result checksum checks passed |
| Overwrite protection | Reaccepting into existing accepted paths raises FileExistsError |
| Real browser | Suzuki 51.35; Toyota filtering and prediction 85.54; held-out prediction 141.46 versus actual 139.00 |
| Mobile | 390 × 844; body/document width 390; no horizontal page overflow |
| Browser logs | No warning/error entries during final checks |

Raw logs: `verification/unittest.txt`, `verification/pip_check.txt`. Machine-readable evidence: `checks.json`, `browser_checks.json`, `source_provenance.json` in that directory.

## Measured primary results

| Model | CV MAE mean | Test MAE | Test RMSE | Test R² | Final fit seconds |
|---|---:|---:|---:|---:|---:|
| Median | 28.3609 | 27.8855 | 44.0346 | -0.0602 | 0.083 |
| Ridge | 8.5247 | 8.4133 | 21.4865 | 0.7476 | 0.203 |
| Random Forest | 5.6002 | 4.9093 | 19.0170 | 0.8023 | 52.429 |
| XGBoost | 6.3770 | 5.9851 | 19.5078 | 0.7919 | 1.753 |
| CatBoost | 6.8716 | 6.5747 | 19.9406 | 0.7826 | 28.656 |

CV used 25.37 minutes of summed fit time. Peak sampled CV process RSS was 302.1 MiB; minimum available system RAM was 1239.1 MiB. These cover sampled training regions, not every rendering/reload operation. Pilot and final-refit measurements are retained separately.

Exact environment versions are in `requirements-lock.txt` and `results/accepted/protocol.json`. All training was local, with sequential searches and two estimator threads.

## Issues resolved

- Preflight string conversion of numeric group hashes changed sorting and the seeded split. Numeric hashes were restored before tuning, reproducing the reviewed partition. The earlier pilot remains separate.
- Initial artifact promotion failed to resolve a relative run path. The incomplete copy was preserved, the promotion-only function was fixed, and acceptance completed. No training, candidate, split, metric, selection or evaluation function changed. Exact source is preserved in `verification/training_before_promotion_fix.py`; raw/normalized hash and AST checks are recorded in `promotion_fix.json` and `source_provenance.json`.
- Browser review found header clipping and a disappearing expander result. Spacing and session state were corrected; updated automated and real-browser checks passed.

Old signatures intentionally reject changed code. The exact source used for the completed run is preserved; fresh standard notebook runs use the current runner and perform actual training. Accepted scientific results were not replaced after these fixes.

## Limits and interpretation

Historical advertisements are not verified transactions, and price scale is unresolved. The 20 primary-test listings above 200 units have MAE 67.02; overall errors are not uniform across price bands. Feature groups are not vehicle identities and near-duplicates may remain. Permutation importance is not causal and correlated inputs complicate interpretation.

Fixed random RF reproduced historical rounded activity scores. Newly fitted random XGBoost differs slightly; the precise cause was not established, and historical scores were not substituted. Different grouped/random test populations prevent attributing their raw score difference solely to overlap or claiming algorithmic improvement.

The dataset had prior exploratory exposure. No external/future-market validation, significance claim or calibrated interval is made. Institution-specific rubric and AI-use rules still require student review; factual acknowledgement is included.

## Deliverables and evidence

- `vehicle_price_assignment.ipynb`: executed experiment, results and discussion.
- `results/accepted/`: protocol, splits, fold scores, predictions, sensitivity, interpretation and figures.
- `models/accepted/`: selected fitted pipeline.
- `results/runs/main-local-20261020-v2/`: completed run and all five model bundles.
- `verification/dashboard-overview.jpg`, `dashboard-desktop.jpg`, `dashboard-mobile.jpg`: actual browser captures.

Dashboard: http://127.0.0.1:8502 while the local server runs. Restart from this directory with `.venv/Scripts/python.exe -m streamlit run app.py`.


## Presentation and submission-readiness review — 23 October 2026

No notebook cell, training, tuning, final evaluation, model export or promotion was executed during this review. All 13 original code-cell sources, execution counts and code-cell metadata were retained. Every original non-stream output (tables, pipeline representations and images) has an identical canonical JSON digest before and after presentation edits.

The tuning cell originally contained 170 fold-progress lines plus the existing selection line; the evaluation cell contained 15 TEST/SENSITIVITY progress lines. Their complete original text was archived in `verification/notebook_progress_original.txt`. The notebook retains the original selection output and every table/chart. This archive is evidence from the earlier experiment, not a new execution log.

Markdown now explains the responsibilities, training boundaries and return values of the main supporting functions. The final academic discussion keeps model comparisons, CV/test separation, importance and limitations. Promotion troubleshooting and resumption/source provenance remain in this verification document instead of the primary discussion. The dataset instructions now specify independent version-3 acquisition and checksum checking.

### Lightweight checks actually run

- 14 non-training tests passed with all five estimator fit methods patched to raise if called; **zero fit calls** occurred. The single existing test that fits a tiny diagnostic Ridge was deliberately not run. This is distinct from the historical 15-test result above.
- Notebook format and Python syntax validation passed without executing notebook cells.
- The retained five-model test table, best-family CV table and ten-row split-sensitivity table match accepted JSON values at their displayed precision.
- All 109 protected source/test/runtime/data/model/result files match the pre-review hashes. In particular, all files under `models/`, `results/accepted/` and `results/runs/` are unchanged.
- All 49 monitored original-activity files still match their earlier checksum snapshot; no original-activity code was executed.
- Every notebook-referenced source module is present: `src/data.py`, `src/models.py`, `src/reporting.py`, `src/training.py`. The inference module and package initializer are also included for the dashboard and saved pipelines.
- `pip check` reported no broken requirements. All 81 installed distributions listed in `requirements-lock.txt` match their pinned versions. No package was installed or changed.
- The verification helper now treats an absent original activity/snapshot as unavailable rather than making that separate project mandatory. Its absent-folder branch was simulated and passed; it does not claim an activity-integrity pass when unavailable.
- Credential-pattern scanning of proposed submission text and secret-filename checks found no suspected credentials. This is a targeted scan, not a guarantee about every possible secret representation.

Evidence: `verification/presentation_before.json`, `presentation_review.json`, `presentation_tests.json`, `presentation_tests.txt`, and `notebook_progress_original.txt`. Older execution and browser logs were retained unchanged. Streamlit's bare-context warnings appeared during AppTest; tests passed. No new real-browser session was needed for this documentation/output-only review; browser evidence above remains dated to the earlier implementation.

### Reproduction and packaging limits

A clean-machine installation, dataset redownload and complete experiment reproduction were **not** performed during this review. Source/import closure, lockfile versions, stored metrics, artifact hashes and non-training inference tests were checked. Normal fresh notebook execution still performs actual training in a new directory. Exact numerical/timing equivalence on another machine is not guaranteed; historical source/signature differences remain recorded above.

Use `SUBMISSION_INVENTORY.md` and `verification/submission_inventory.json` as the proposed file allowlist. Include the complete successful source run because the existing overwrite-protection test reads its completion manifest and all five model bundles. The repeated selected RF model in that run is therefore required by the current test arrangement; its accepted deployment copy is also retained. The preliminary pilot and incomplete promotion copy are excluded from the proposed package, but were not deleted or modified locally.

The environment, pip/IPython/Matplotlib caches and bytecode directories are generated development files and should not be submitted. The raw CSV and download ZIP are excluded by default; include acquisition instructions instead unless redistribution permission and institutional requirements support including data.

Remaining institutional checks: confirm the upload/archive limit, mandatory report/cover sheet/student identification/declaration, and AI-use policy. If grading must run offline, arrange legitimate access to the checksum-matching CSV. Currency scale, marketplace rights and advertised-versus-transaction limitations remain unresolved research constraints, not presentation edits. No submission archive was created and nothing was committed, pushed, published, deployed or submitted.
