# Dataset acquisition and provenance

Author: Prasad Nirmal. Dataset: [SriLankan Vehicle(Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset). Required version: **3**.

## Obtain the file independently

On Kaggle, select version 3 and download its archive. Extract `car_price_dataset.csv` directly into this assignment's `data/` directory. A Kaggle account may be required. Do not substitute the current/latest version without checking its hash.

Alternatively, from PowerShell in a **fresh assignment copy**:

```powershell
New-Item -ItemType Directory -Path data -Force | Out-Null
Invoke-WebRequest -Uri 'https://www.kaggle.com/api/v1/datasets/download/prasadnirmal/srilankan-second-vehiclecar-price-dataset?datasetVersionNumber=3' -OutFile data/dataset-v3.zip
Expand-Archive -LiteralPath data/dataset-v3.zip -DestinationPath data
(Get-FileHash -LiteralPath data/car_price_dataset.csv -Algorithm SHA256).Hash
```

If the endpoint redirects to sign-in, is unavailable or yields a different file, use Kaggle's version-3 download interface. Do not bypass terms or silently use another version. The endpoint was not redownloaded during the presentation review; local checksum verification passed.

Expected CSV SHA256:

`4208d2673cf4d9c8d3d6042dc8a1b97004671812441975f5879550059fbce1ab`

Expected raw shape: 9,788 rows × 17 columns. `src/data.py` verifies the checksum and schema before cleaning. No access to the original activity is required. The local development copy was originally copied from that read-only activity and remains unchanged.

## Attribution and distribution

The earlier activity records ikman.lk and Riyasewana.com as marketplace sources and an MIT Kaggle licence. Current metadata and underlying marketplace redistribution rights could not be independently established. Keep attribution and check rights before redistributing the CSV. Raw data is excluded from Git and the proposed submission by default; include these acquisition instructions instead.

The stored Price scale is not conclusively established, so predictions and metrics use original dataset units. Mileage is derived from year and excluded as a predictor.
