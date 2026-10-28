# GitHub publication preparation — 28 October 2026

The student supplied the current Word report after the writing review. Its cover/ending edits are preserved; it now ends with References. The missing PDF was regenerated from a byte-identical copy without changing the DOCX. Tables and numerical findings match the prior reviewed report. AI assistance remains acknowledged in the repository README. Earlier report-review descriptions below apply to the versions reviewed at those times.

The approved GitHub destination is https://github.com/yulepaulusha-uom/sri-lanka-vehicle-price-ml. Azure account connection, resource creation and public application deployment remain unapproved. The workflow is manual and defaults to no deployment. No scientific model, split, result or notebook is changed for this publication.

---

# Final report and deployment preparation — 27 October 2026

This is local preparation only. No commit/push, account connection, Azure provisioning or public deployment has occurred. Earlier dated sections below retain the historical experiment/EBM execution evidence.

## Report review

The previous DOCX/PDF are backed up in `verification/deployment/before/report/`. The entire report was read and 25 paragraphs refined for direct MSc-level writing. Introduction/EBM justification and comparison with conventional boosting are clearer; resource-monitoring detail is shorter. All tables and embedded figures compare exactly with the backup. Equations, citations, measured comparisons, negative estimates, sparse-category/expensive-vehicle limitations, ethics and the previously inspected holdout remain.

The final PDF is ten pages (cover, eight main pages, references/disclosure). All pages were visually reviewed; the last introduction wording change was rerendered and page 2 reinspected. The mileage equation is visible on page 2 and additive equation on page 3. The honest OpenAI Codex acknowledgement is retained, with its temporary instruction to adapt wording removed. University academic-integrity/AI-use policy was not supplied: student review against the actual policy remains necessary; compliance is not claimed.

## Scientific preservation

The preparation baseline covers **146 scientific/source/model/result/dependency/notebook files**; all final SHA256 values match. Both academic notebooks are unchanged. The original activity's 49 recorded hashes also match. No model was trained, retuned, converted or overwritten. Existing RF/EBM preprocessing is present in the saved artifacts.

Only the UI import changed for deployment: a new `src/deployment_runtime.py` normalizes Windows manifest separators in memory. The immutable research loader/model manifest remains untouched. The runtime app uses actual saved RF/EBM and recorded results/figures without the raw dataset.

## Clean Linux verification

An official `python:3.12-slim-bookworm` container, CPython 3.12.16, was used solely for local verification. It was limited to two CPUs and 1.5 GiB RAM, with the runtime source/models mounted read-only and a loopback-only published port. All **45 exact runtime packages installed from binary wheels**; pip check passed. XGBoost, CatBoost, matplotlib, seaborn and Jupyter are excluded from production. The protected original environments were not changed.

**Six Linux checks passed with zero fits**, covering both saved models/preprocessing, expected held-out predictions, additive identity, portable paths, invalid/unseen inputs, UI brand/model selection, recorded metrics, chart generation, no raw CSV/training dependencies and a real negative-output warning. The current Windows research/regression checks also passed: **34 tests, zero failures/errors/skips, zero fits**. The original small Ridge diagnostic fit remains excluded.

A real browser on the Linux server produced RF 51.35 and EBM 76.19 for the default Suzuki form; EBM signed sum 76.186059 differed by 2.8e-14. Sparse-model warning and all seven static images loaded. The EBM results tab displayed MAE 9.764, RMSE 23.908, R² 0.687 and the 38-negative-estimate warning. Browser console query returned no errors/warnings. Local restart health recovered from HTTP 200 to HTTP 200 in 2.284 seconds, and browser predictions worked after restart. This does not verify Azure's managed restart/proxy.

The source runtime directory is 62 files/about 22.51 MiB, plus its checksum index. Linux installation logs, model/UI test output, report review and protected before/after evidence are retained locally in `verification/deployment/`. A shareable summary is `deployment/verification.json`.

## Workflow and approval boundary

The workflow was statically parsed using a temporary isolated YAML parser. An initial unquoted pip-flag scalar was corrected to a block scalar. Action uses are pinned to commit SHAs; the workflow is manual, deploy defaults to false, an explicit repository enable flag is required, and deployment uses a production environment gate with OIDC. The workflow has not been executed on GitHub.

Proposed configuration/costs and exact runtime pins are in `deployment/README.md`, `azure-settings.proposed.json` and `requirements.txt`. Public retail API price for proposed Southeast Asia Linux B1 is USD .018/hour (USD 13.14/730 hours), excluding other charges and without a subscription quote. Region/quota, global app-name availability, actual subscription runtime list, budget and OIDC permissions remain pending. No account was accessed to assume these.

GitHub publication is prepared as a separate curated file list/directory, with no .git initialization, commit or remote. It excludes raw data, environments, caches, private machine-profile paths and raw local logs. Detailed local academic inventory and curated GitHub contents differ intentionally; use `deployment/github-files.json` for repository publication. Archival preservation records omitted from that curated list are reported unavailable when absent rather than misreported as checked.

The next external steps require explicit user approval: repository contents/visibility and URL; then Azure subscription/region/B1 budget/names and secure identity settings. The public HTTPS URL must be tested after approved deployment before claiming success. The final ZIP, student-recorded 3–5-minute video and Moodle upload remain uncreated.

---

# EBM extension verification — 27 October 2026

This section records the new subsequent study. Earlier verification sections below describe the historical five-model work and presentation review; their statements about no retraining apply to those dated operations.

## Integrity and provenance

`verification/ebm_extension/integrity_before.json` and `integrity_after.json` verify **166 immutable assignment files and 49 original activity files unchanged**. The initial broad ledger also included the authorized mutable submission inventory index; its previous bytes are retained in historical_copies. The final strict comparison excludes that explicitly authorized document revision and verifies all 166 immutable assignment files plus 49 original-activity files. The original completed notebook, accepted metrics/splits/models, old requirements and original environment are preserved. `report/archive/` retains both comparative report formats; `verification/ebm_extension/historical_copies/` retains earlier UI/document revisions. All changes occur within the assignment directory.

## Actual extension execution

- Isolated Python 3.12.10 environment; interpret-core 0.7.8, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1. Other packages retain the original lock versions. A fresh isolated installation and pip check passed; exact extension lock included.
- Native API signatures, categorical missing/unseen behavior, explicit bag weights and export/reload passed in the 2,000-row development pilot: fit 0.9217 s, sampled peak RSS 156.06 MiB.
- Saved development/holdout rows and five outer folds reused exactly. Four explicit internal bags preserve groups, reserving 15% of groups for validation with seeds 42–45 inside the current training partition. All 31 recorded CV/final bag matrices are checked for separation and exact row IDs. No one-hot encoding or global fitted preprocessing is used.
- Six configurations, 30 CV fits, summed measured fit time 149.4723 s, peak sampled process RSS 204.93 MiB. All fold metrics/resources/stopping counters are saved. Five of 120 CV bag counters reach the cyclic cap; one final bag also reaches it.
- Initial notebook execution stopped after three saved folds at 504.77 MiB available RAM, below 512 MiB. The user closed apps and authorized resumption. The notebook reused those fits and completed the remaining 27, with resumed minimum available RAM 1,012.18 MiB. No search expansion or method/safeguard change occurred.
- C4 selected by grouped CV MAE 10.301987 ± 0.713360 sample SD: bins 256, rate .02, leaf support 20. Selection was frozen before final fit/scoring.
- Final EBM fit 7.870117 s; serialized model 147,531 bytes; SHA256 `b21f4a37aabf473a405e12145bca5d011a544e17b415901cd04272779c364759`.
- Holdout MAE 9.764403, RMSE 23.907896, R² .687464. Reload predictions match. Thirty-eight negative raw estimates remain in metrics; neither target nor output was transformed/clipped. The holdout was previously inspected and is not described as independent untouched evidence.
- `complete.json` seals 90 extension evidence/model files and five supporting EBM source modules. Completed-run inspection is read-only; a fresh run ID uses separate directories and leaves the reviewed model pointer in place.

## Scientific and UI verification

`final_checks.txt` records **34 tests passed, 0 failures, 0 errors, 0 skips, 0 fit calls**. The runner guards fit methods for all six algorithms and excludes the unchanged original small Ridge diagnostic fit test. Tests cover original behavior, saved outer and internal groups, schema/category handling, invalid numeric input, export/reload, repeated inference, native global structure, exact local sums, negative-output behavior, actual negative frontend output, sealed-run reuse, fresh-run paths and protected hashes. External activity checks are explicitly skipped if the separate original directory is absent on another machine. Historical preliminary-run files, partial-promotion copies and the original environment configuration are optional local preservation evidence: checked if present and reported unavailable in a clean package, not required submission dependencies.

The local Audi example was predeclared as cleaned row 11. EBM predicts 149.404204 rather than copying RF 141.456916. Intercept plus all terms differs from prediction by 2.84e-14. Year, Brand and engine are the largest native terms; effect plots include development support. RF permutation evidence is preserved and interpreted as a different quantity.

Real browser verification checked form submission, EBM waterfall/sum, rare-category warning, fixed holdout outputs, separately recorded EBM metrics and negative-count disclosure. A 390×844 viewport had document width 390 with no page overflow. Screenshots and observed values are recorded in `browser_checks.json`; browser log query returned no error/warning entries. The browser tool emitted a Statsig initialization notice outside that app log query. Viewport was restored. The temporary screenshot receiver was stopped; Streamlit remains a local-only demonstration.

The final notebook has actual executed outputs and no error cells. Its training execution is archived as `notebook_training_execution.ipynb`; the resource-stop notebook/log are also retained. The later presentation verification actually re-executed notebook cells under an EBM fit guard, loading completed checkpoints; it did not retrain or rescore for selection. `notebook_verification_execution.log` records completion. Jupyter emitted Windows selector/plain-TCP warnings during local execution; these were nonfatal and remain in raw logs.

## Report and readiness

The revised editable report and PDF have ten pages, including eight main-content pages. All page images were visually reviewed; after wording/citation revisions, three changed pages were reinspected and seven were byte-identical. The mileage formula is visible on page 2 and additive equation on page 3. `report_checks.json` records final hashes and checks.

The packaged render script failed because LibreOffice was unavailable; this was diagnosed in `report_renderer.log`. Installed ONLYOFFICE with existing font cache converted successfully, followed by bundled Poppler rasterization. No new desktop renderer was installed. Historical report copies remain unchanged.

`SUBMISSION_INVENTORY.md` and its JSON are the packaging allowlist. Environments, caches, raw CSV, obsolete duplicate reports, development edit helpers and preview renderer files are excluded. Model/evidence files are retained. No archive, video, commit, push, deployment or upload was made. Student review, formal eligibility uncertainty, the required student-recorded video, institutional limits/declarations and redistribution rights remain submission concerns.

## Clean submission copy check

An exact hash-verified copy of the proposed source/model/evidence payload was tested with only the locally checksum-verified CSV added. It had no original activity folder, original virtual environment or development caches. **32 checks passed and two external original-activity checks were explicitly skipped; zero model fits, failures or errors occurred.** This verifies package independence using the installed compatible extension interpreter, not a second fresh dependency installation or new Kaggle download.

The first copy used an earlier allowlist and missed `check_integrity.py`, causing two import errors. That evidence is preserved. The finalized allowlist includes the helper; the corrected copy was verified against every listed hash before tests passed. `submission_smoke.json` and `submission_smoke_tests.txt` record the final outcome. The temporary smoke copy is excluded from submission.

---

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


