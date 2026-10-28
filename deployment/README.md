# Azure App Service preparation

Status: GitHub publication approved to https://github.com/yulepaulusha-uom/sri-lanka-vehicle-price-ml. Azure account connection, resource provisioning and public application deployment remain unapproved and have not been performed. The reference PDF was not found in the attachment/workspace search, so this uses the requested GitHub → GitHub Actions → Linux Python App Service approach and current official documentation.

## Proposed resources for approval

| Setting | Proposal |
|---|---|
| GitHub repository | `yulepaulusha-uom/sri-lanka-vehicle-price-ml`, separate from the original activity repository |
| Resource group | `rg-it5514-269112b` |
| App Service plan | `asp-it5514-269112b-b1`, Linux B1, one instance |
| Web App | `it5514-269112b-vehicle-price` (global name availability not checked) |
| Region | `southeastasia` (Singapore), subject to the subscription's allowed regions/quota |
| Publish | Code, built-in Linux Python stack; no production Docker container |
| Python | 3.12; clean Linux test used CPython 3.12.16 |
| Startup | `python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true` |

Python 3.12 is documented for Linux App Service. Microsoft updates patch versions, so the Azure patch need not equal the historical Windows 3.12.10 patch. The exact subscription runtime list must be rechecked with `az webapp list-runtimes --os linux` or the portal after account access is approved. No Azure CLI is installed locally and no subscription has been queried. Region availability and name availability remain unverified. [Python runtime support](https://azure.github.io/AppService/2023/11/14/Node-20-Python-312-available-on-app-service.html), [patch update policy](https://github.com/Azure/app-service-linux-docs/blob/master/Runtime_Support/python_support.md).

B1 provides one core and 1.75 GB RAM; it is the smallest proposed dedicated tier with enough measured headroom for this low-traffic demonstration. This is a viability proposal, not load-testing evidence. Free-tier memory, CPU quotas and connection limits are not assumed sufficient. The local Linux app used about 293 MiB at one idle sample after verification; real traffic and platform overhead can increase this.

The public Azure Retail Prices API returned Linux B1 in Southeast Asia at **USD 0.018/hour**, or **USD 13.14 for 730 hours**, on this review. This is the base plan estimate, not a subscription quote. Tax, outbound traffic, logging/monitoring, domain costs and GitHub Actions limits may add charges. Discounts/credits depend on the account. A paid plan can continue accruing charges even if its app is stopped; deleting the approved dedicated plan/resource group after the demo is the clearest way to avoid ongoing allocation. Do not provision until budget/region are explicitly approved. [Azure pricing](https://azure.microsoft.com/en-us/pricing/details/app-service/linux/), [public retail API](https://prices.azure.com/api/retail/prices?%24filter=serviceName%20eq%20%27Azure%20App%20Service%27%20and%20armRegionName%20eq%20%27southeastasia%27).

## Runtime files and dependencies

`prepare.py` creates a new `build/azure/` directory. It refuses an existing destination. It copies the real root `app.py`, inference source, fitted RF pipeline/preprocessing, fitted EBM, verified metadata/results and required figures. It excludes the raw CSV, environments, notebooks, reports, alternate fitted models and training checkpoints. It copies this folder's **45 exact runtime pins** to the payload's root `requirements.txt`; the historical root dependency files remain unchanged. The runtime folder has 62 files, about 22.51 MiB before its manifest.

Core versions: Streamlit 1.65.0, NumPy 2.1.3, pandas 2.2.3, SciPy 1.18.1, scikit-learn 1.6.1, joblib 1.6.0, interpret-core 0.7.8 and Plotly 7.1.0. Every transitive runtime dependency is pinned in `requirements.txt`. XGBoost, CatBoost, matplotlib, seaborn, Jupyter and training utilities are unnecessary for the two loaded models and are excluded. Scientific model files/formats are unchanged.

The RF pipeline stores fitted imputation/one-hot preprocessing. EBM stores native fitted bins. `src/deployment_runtime.py` normalizes Windows separators in the immutable EBM manifest in memory; it never edits that manifest. Paths resolve relative to the application root. Both models verify SHA256 before loading. No raw CSV is read at startup or prediction.

Local runtime-only setup:

```powershell
py -3.12 -m venv .venv-runtime
.\.venv-runtime\Scripts\python.exe -m pip install -r deployment/requirements.txt
.\.venv-runtime\Scripts\python.exe deployment/prepare.py
.\.venv-runtime\Scripts\python.exe deployment/smoke_test.py --app-root build/azure --cases deployment/smoke_cases.json --output build/runtime-checks.json
# To run the prepared payload locally, change into build/azure and use the same interpreter.
```

Research reproduction still uses `requirements-ebm-lock.txt`, the notebooks and acquisition instructions. It must not overwrite the accepted runs. The notebooks and evaluation findings were not changed during this deployment preparation.

## Build, startup and proxy settings

For source-folder deployment through `azure/webapps-deploy`, set **SCM_DO_BUILD_DURING_DEPLOYMENT=true** before deployment. Oryx then installs only the runtime requirements at the payload root. Do not deploy the repository root or the Windows virtual environment. No extra requirements are installed during startup. Do not set `WEBSITE_RUN_FROM_PACKAGE` for this remote-build approach. [Python build documentation](https://learn.microsoft.com/en-us/azure/app-service/configure-language-python).

The production startup overrides the local 127.0.0.1:8502 configuration and binds 0.0.0.0:8000. This port follows the built-in Python App Service pattern. `WEBSITES_PORT` is a custom-container routing setting and is not required for this code-stack design. The production Streamlit config retains CORS/XSRF protections and disables usage telemetry. If a reverse-proxy issue occurs, inspect logs and configure the known public host; do not permanently disable protections as a workaround. [Python startup](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python), [Streamlit proxy troubleshooting](https://docs.streamlit.io/knowledge-base/deploy/remote-start).

Linux App Service supports WebSockets automatically; `webSocketsEnabled` is not an applicable Linux switch. Configure Always On on the approved B1 plan, HTTPS-only access, and the health-check path `/_stcore/health`. Suggested warmup settings are `WEBSITE_WARMUP_PATH=/_stcore/health` and `WEBSITE_WARMUP_STATUSES=200`. HTTP health confirms the server process, not successful model inference, so also test actual browser prediction after deployment. Start with one instance; no database, backend service, custom domain or scale-out is required. [Linux WebSockets/warmup FAQ](https://learn.microsoft.com/en-us/troubleshoot/azure/app-service/faqs-app-service-linux-new).

After an approved deployment, inspect Oryx/build logs, Streamlit startup logs and Azure log stream. Test restart recovery and then repeat a real RF/EBM form submission, explanation sum, metrics and negative warning at the public HTTPS URL. Public-proxy and managed restart behavior cannot be proven by the local container; those remain post-approval checks.

## GitHub and OIDC setup after approval

1. Use the approved repository https://github.com/yulepaulusha-uom/sri-lanka-vehicle-price-ml. Keep the original local class activity directory strictly read-only; publication uses the separate prepared assignment directory.
2. The approved initial GitHub publication uses the curated `github-files.json` content, excluding raw data, environments, caches and local-only logs. Future changes should receive their own review.
3. Confirm Azure subscription, permitted region, B1 budget, names and access permissions. Provision only after explicit approval. Check Python 3.12 availability, plan quota and global Web App name first.
4. Create an Entra application/service principal or suitable managed identity with **Website Contributor scoped to this Web App**. Configure a federated credential for issuer `https://token.actions.githubusercontent.com`, audience `api://AzureADTokenExchange`, subject `repo:yulepaulusha-uom/sri-lanka-vehicle-price-ml:environment:production`. This subject must match the approved repository and production environment exactly. No client secret/publish profile is needed. [OIDC setup](https://learn.microsoft.com/en-us/azure/app-service/deploy-github-actions).
5. In GitHub, create the `production` environment and require a reviewer if the account/plan supports that control. Add Azure IDs as GitHub environment secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`; do not put credentials in files. Set repository variable `AZURE_WEBAPP_NAME`. Keep `AZURE_DEPLOY_ENABLED` unset/false until approval.
6. The commit-pinned workflow runs only by **workflow_dispatch**. Its deploy input defaults to false. It verifies Linux inference first, and deploys only when deploy=true, AZURE_DEPLOY_ENABLED=true, branch=main and the production environment gate permits it. No automatic deployment on push is configured. Action commit hashes and release provenance are in `action_versions.json`.
7. After explicit approval, commit/push the reviewed new repository, configure the approved resources/settings/OIDC, then run verification-only first. Enable and run deployment only after that passes. Verify the public URL in a browser before calling deployment successful.

## Academic review and submission

The revised report is still local-demonstration based; it makes no Azure deployment claim. AI assistance is acknowledged in the repository README. The university's actual academic-integrity/AI-use policy was not provided, so its wording needs student review against that policy; institutional compliance is not claimed. Source attribution, prior holdout exposure, negative predictions, sparse categories and expensive-vehicle errors remain explicit.

No final source ZIP, video or Moodle upload has been created. Required student actions remain review/approval, repository URL, Azure choices and budget, policy/declaration review, and the student-recorded 3–5-minute demonstration video.
