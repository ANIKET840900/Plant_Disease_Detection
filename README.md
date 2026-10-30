# Plant Disease Detection and Affected-Area Estimation

CPU-first full-stack project for image-based plant disease classification and visible lesion-area estimation.

## Current model status

The MobileNetV3-Small classifier was trained on 54,305 PlantVillage images covering 38 labels. Its saved internal validation accuracy is 94.14%; this is not an independent field-performance estimate, and the score is not calibrated. The classifier checkpoint is in `saved_models/`. A trained leaf/lesion segmentation checkpoint is not available, so exact lesion masks and affected-area percentages are not produced.

## Architecture

```text
React + Vite frontend
        |
        | HTTP/JSON and image uploads
        v
FastAPI backend
  |-- image quality and preprocessing
  |-- MobileNetV3 classification
  |-- overlapping image-window screening
  |-- candidate crop and original-image overlay generation
  |-- conservative diagnostic guidance and source links
        |
        v
Local datasets, checkpoints, visualizations, and reports
```

## Project layout

```text
backend/
  app/
    preprocessing/ schemas/ services/ segmentation/
    config.py
    main.py
  tests/
  requirements.txt
frontend/
  src/
    App.jsx
    styles.css
  package.json
  vite.config.js
training/
data/
  raw/ processed/ masks/
saved_models/
reports/
scripts/
```

## Requirements

- Windows laptop with Python 3.11 and Node.js
- CPU-only execution is supported by design
- No GPU, cloud service, paid API, or LLM is required
- Datasets and trained model checkpoints are supplied separately and are not committed

## Windows PowerShell setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

Set-Location frontend
npm install
Set-Location ..
```

If PowerShell blocks activation for the current user, use the one-time command below in an elevated PowerShell window:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run the Phase 1 checks

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest backend\tests
python -m pytest training\test_prepare_splits.py
python -m compileall backend\app
```

Start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload
```

Start the frontend in a second PowerShell window:

```powershell
Set-Location frontend
npm run dev
```

The backend health endpoint is `http://127.0.0.1:8000/health`. The frontend runs at the URL printed by Vite.

## Reproducible classifier training and evaluation

The repository includes `training\config.yaml` (JSON syntax, valid YAML 1.2) as the shared training configuration. `prepare_dataset.py` and the training workflow use `data\processed\PlantVillage`. On Windows, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m training.prepare_splits `
  --data data\processed\PlantVillage `
  --output reports\generated\split_manifest.json `
  --seed 42

.\.venv\Scripts\python.exe -m training.fetch_plantdoc

.\.venv\Scripts\python.exe -m training.prepare_combined_manifest `
  --plantvillage data\processed\PlantVillage `
  --plantvillage-manifest reports\generated\split_manifest.json `
  --plantdoc data\raw\PlantDoc_dataset `
  --output reports\generated\combined_split_manifest.json

.\.venv\Scripts\python.exe train_classifier.py `
  --manifest reports\generated\combined_split_manifest.json `
  --model-dir saved_models\candidate_combined

.\.venv\Scripts\python.exe -m training.evaluate_classifier `
  --manifest reports\generated\combined_split_manifest.json `
  --model-dir saved_models\candidate_combined `
  --baseline-dir saved_models `
  --report-dir reports\generated\candidate_combined_evaluation

.\.venv\Scripts\python.exe -m training.evaluate_plantdoc `
  --data data\raw\PlantDoc_dataset `
  --model-dir saved_models\candidate_combined `
  --report reports\generated\candidate_combined_plantdoc_evaluation.json
```

The preparation pass validates every supported image, reports unreadable files, removes exact duplicate copies from the eligible sample set (without deleting user data), excludes exact duplicates found under conflicting labels, groups near-duplicate images so they cannot cross partitions, and writes deterministic stratified train/validation/calibration/test assignments. The resulting manifest and counts are saved under `reports\generated\`. Near-duplicate image groups are retained in a single split; this avoids evaluation leakage without discarding distinct captures.

The combined manifest adds only PlantDoc's official train partition, after checking exact and perceptual-hash duplicates against every PlantVillage partition and within the PlantDoc training images. The PlantVillage validation, calibration, and test partitions are not augmented with PlantDoc. PlantDoc's official test partition is kept out of training and evaluated separately. Training starts from torchvision ImageNet MobileNetV3-Small weights, sets head dropout to 0.3, uses field-style augmentation on training examples only, and applies square-root inverse-frequency class weights plus label smoothing. It freezes the complete feature extractor for three head-only warmup epochs, then unfreezes the full backbone and fine-tunes backbone/head groups at separate learning rates with linear warmup and cosine decay. Early stopping is based on validation loss (patience 5). Training logs and candidate weights are saved under the selected candidate directory; the production model is not overwritten during training. Deterministic algorithms and seed 42 are enabled.

Evaluation fits temperature scaling and a confidence referral threshold on the calibration partition only. It reports test accuracy, macro precision/recall/F1, balanced accuracy, per-class metrics/support, confusion matrix, calibration error, Brier score, negative log-likelihood, and misclassified image paths. Training stores a hash of the exact split manifest and the name/count of the training partition. The evaluator marks test scores exploratory when checkpoint provenance is missing or mismatched and blocks promotion unless candidate and baseline split provenance both match. This prevents a legacy checkpoint with an undocumented training split from being presented as an independent test or promoted via an unverifiable comparison.

The PlantVillage split is not field-photo validation. The training workflow does not claim plant/session-level grouping because these identifiers are not present. Duplicate-aware grouping is not a substitute for plant/session grouping. Do not claim validated out-of-distribution detection, lesion segmentation, or affected-area percentages from PlantVillage results. An unrecognized input is not reliably detectable merely because its softmax confidence is low. Use the interface's review/uncertainty notice and seek expert confirmation for uncertain or out-of-scope plants.

## External PlantDoc evaluation

The project can prepare the Cropped-PlantDoc dataset's official train/test files from the upstream Git repository without checking out its filename containing `?` (which is invalid on Windows):

```powershell
.\.venv\Scripts\python.exe -m training.fetch_plantdoc
```

The fetcher retains the upstream `README.md` and `LICENSE.txt`, sanitizes invalid Windows filename characters, verifies image readability, and writes a source manifest containing original labels and official split names. Dataset terms are CC BY 4.0; preserve attribution. The evaluator maps only the documented PlantDoc folder labels to the classifier's known PlantVillage labels and reports unsupported/corrupt counts:

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_plantdoc `
  --data data\raw\PlantDoc_dataset `
  --model-dir saved_models `
  --report reports\generated\plantdoc_evaluation.json
```

PlantDoc's images are internet-collected natural-background examples, not a prospectively sampled farm/session cohort. The external test result is a cross-dataset stress test, not definitive field generalization or validated OOD assessment. Baseline promotion is blocked unless both checkpoints have verified provenance for the same split manifest.

The dataset paper is Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra, “PlantDoc: A Dataset for Visual Plant Disease Detection” (2020), [ACM DOI 10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196), [arXiv:1911.10317](https://arxiv.org/abs/1911.10317). The repository and license are at [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset).

## Single-leaf and whole-plant screening

In **New analysis**, choose **Single leaf** for a close-up or **Whole plant** for an image showing multiple leaves/the full plant. Choose **Use webcam to capture** to grant camera access, frame the plant in the live preview, and select **Capture photo**. The captured JPEG is treated like a regular upload and can be screened and reopened from History. Camera access is requested only after selecting this option; the live stream stays in the browser and is stopped after capture, when turned off, or when leaving the analysis view. Webcam access requires localhost or HTTPS and browser permission. The backend screens overlapping square image windows and returns only crops whose top classifier label is non-healthy and whose uncalibrated score is at least 0.60. It also returns the original image with candidate crop boxes outlined.

After screening, select **Print / Save report as PDF** to open the browser print dialog. Choose **Save as PDF** to export a report containing the uploaded filename, timestamp, screening mode, original image with candidate locations, each returned crop and its model output, safety guidance, limitations, and institutional source links.

For supported MobileNetV3-Small checkpoints, the report also includes a Grad-CAM attribution overlay for the overall single-leaf prediction or the highest-scoring whole-plant candidate. The overlay highlights image features influencing a class score; it is not a verified lesion mask, causal explanation, or affected-area estimate. The current dataset has no lesion masks or field validation with which to verify that highlighted features correspond to disease symptoms rather than other image cues.

The **History** view saves every completed analysis without a fixed 20-entry limit. Each entry, including the full result and original image when available, is stored in IndexedDB so it can be reopened later in the same browser profile. Older summary-only entries are migrated from local storage and explain when the original image is unavailable. Use **Delete this analysis** to remove one entry or **Delete all history** to clear the browser's saved reports. History remains local to that browser profile and is not synced to other devices; browser storage limits and user-cleared site data can still remove saved records.

The returned crops are **candidate image regions**, not exact diseased pixels. The PlantVillage classifier was trained on leaf images, not whole-plant scenes. Do not interpret its score as a calibrated probability, and do not interpret an empty candidate list as proof that the plant is healthy. The interface describes symptom evidence and severity as unverified/unassessed where the model cannot establish them.

The guidance is limited to cautious first steps and directs users to local diagnostic professionals. It intentionally does not select biological products or pesticides from an image prediction. The general institutional references displayed in results are:

- [American Phytopathological Society Education Center](https://www.apsnet.org/edcenter/Pages/default.aspx)
- [FAO Plant Production and Protection](https://www.fao.org/plant-production-protection/en/)
- [USDA APHIS Plant Pests and Diseases](https://www.aphis.usda.gov/plant-pests-diseases)

These are general diagnostic resources, not disease-specific treatment trials. No individual scientist or publication author is attributed unless a disease-specific, verifiable source is added.

The screening API is `POST /screen-plant`, with multipart fields `file` and optional `mode` (`single_leaf` or `whole_plant`; defaults to `whole_plant`). Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/screen-plant" `
  -F "mode=single_leaf" `
  -F "file=@C:\path\to\leaf.jpg"
```

## Development phases

1. Environment setup and project skeleton
2. Dataset preparation and MobileNetV3 classifier
3. U-Net segmentation and mask validation
4. Affected-area calculation and unit tests
5. Confidence calibration, image-quality checks, and referral policy
6. FastAPI endpoints
7. React dashboard and API integration
8. SQLite history and PDF reports
9. End-to-end testing and performance evaluation
10. Documentation and patent-oriented research

## Data and model policy

Classification labels and image-window crops are not segmentation masks. A pixel-level annotated dataset or manually created masks are required for segmentation training and exact lesion localization. The application does not fabricate masks, treatment trial results, or diagnoses.

Results are automated image-based assessments, not confirmed agricultural diagnoses.
