# EBM extension completion status

The four approved phases are complete. This is a subsequent EBM study; the historical RF remains its original accuracy-selected winner. No historical algorithm was refitted.

## A Preservation environment and pilot

- 166 immutable assignment hashes and 49 original-activity hashes match before/after.
- Historical notebook, five-model artifacts, metrics, splits, dependency files and environment remain unchanged.
- Prior UI/document files and both comparative report formats are archived.
- Python 3.12.10 isolated `.venv-ebm`; interpret-core 0.7.8. Fresh installation and pip check passed.
- Development pilot: 2,000 rows, 200-round cap, 0.9217 seconds, 156.06 MiB sampled peak RSS; export/reload and missing/unseen native inputs passed.

## B Experiment and XAI

- Six configurations were declared before tuning; 30 grouped outer fits completed.
- Initial execution checkpointed three folds then stopped at 504.77 MiB available RAM. After the user closed apps and confirmed, the same notebook reused those checkpoints and ran the remaining 27. Resumed fits kept at least 1,012.18 MiB available. No protocol or safeguard was changed.
- Selected C4: bins 256, rate 0.02, min leaf support 20. CV MAE 10.301987, sample SD 0.713360.
- Frozen selection preceded final fit and one holdout evaluation. The holdout had already been inspected historically and is explicitly disclosed as such.
- Test MAE 9.764403, RMSE 23.907896, R² 0.687464; final fit 7.870117 seconds; model 147,531 bytes.
- EBM did not beat Ridge or historical tree ensembles. It produced 38 negative raw holdout estimates, retained in evaluation and flagged in UI. No clipping was introduced.
- All native global/year/engine/supported-Brand plots use the actual model and development support.
- Local Audi prediction 149.404204 = intercept 53.441960 + signed terms 95.962244, numerical difference 2.84e-14. RF 141.456916 remains a distinct historical output.
- Sealed evidence/model hashes are in `results/ebm_extension/runs/ebm-local-v1/complete.json`. Ordinary completed-run reuse does not refit or rewrite these records.

## C Notebook and frontend

- New notebook executes reusable training, tuning, export/evaluation and XAI functions. Its actual training execution and earlier interruption are preserved separately.
- Later notebook presentation verification used cached evidence with EBM fitting forbidden; it did not repeat the search.
- Existing Streamlit design and RF capability retained. New EBM prediction/waterfall, all-term table, rare-input warnings and separate metrics passed real-browser checks.
- Browser form: RF 51.35 and EBM 76.19 for the default Suzuki input; exact EBM sum 76.186059. Heldout demo: RF 141.46, EBM 149.40, actual 139.00.
- Desktop screenshot and 390×844 mobile screenshot saved. Mobile document width equals viewport width; browser error/warning log query returned no entries. Viewport restored.
- **34 guarded tests passed, zero failures/errors/skips, zero fit calls** in the final run. The original small diagnostic Ridge fit test is excluded; original test source is unchanged. Exact logs are `final_checks.txt` and `all_checks.json`.

## D Report and submission documentation

- Updated report DOCX/PDF has ten pages: cover, eight main pages, references/disclosure.
- All pages visually inspected. Mileage formula is visible on page 2 and additive formula on page 3. Final three edited pages were reinspected; remaining seven are byte-identical to the first reviewed render.
- Packaged renderer failed because LibreOffice is unavailable. Existing ONLYOFFICE conversion with its font cache and bundled Poppler succeeded; both diagnostic logs are retained. No desktop LibreOffice was installed or used.
- README, VERIFICATION and exact proposed submission inventory updated. Fresh reproduction uses a new EBM_RUN_ID and does not silently change the dashboard model.
- Dataset acquisition/attribution instructions and compatible dependency locks included. Dataset checksum verified, download not repeated.

## Remaining decisions

Student review remains necessary, including understanding the algorithm and accepting the eligibility interpretation. Formal lecturer eligibility approval is not claimed. Record the required 3–5-minute video after review. Confirm institutional upload limits, any cover/declaration/AI-use wording and dataset redistribution rights before packaging. No ZIP/video/commit/push/deployment/Moodle upload has been performed.

## Clean submission copy check

An exact hash-verified copy of the proposed source/model/evidence payload was tested with only the locally checksum-verified CSV added. It had no original activity folder, original virtual environment or development caches. **32 checks passed and two external original-activity checks were explicitly skipped; zero model fits, failures or errors occurred.** This verifies package independence using the installed compatible extension interpreter, not a second fresh dependency installation or new Kaggle download.

The first copy used an earlier allowlist and missed `check_integrity.py`, causing two import errors. That evidence is preserved. The finalized allowlist includes the helper; the corrected copy was verified against every listed hash before tests passed. `submission_smoke.json` and `submission_smoke_tests.txt` record the final outcome. The temporary smoke copy is excluded from submission.
