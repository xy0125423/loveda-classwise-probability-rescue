# Reproduction guide

Run commands from the repository root in a fresh environment. Install `python -m pip install -e ".[full,dev]"`. Config files use the JSON subset of YAML, parsed by the standard library. No paths to internal research folders are needed. CPU is the documented training/inference device.

## 1. Obtain local inputs

Get LoveDA following dataset.md. Obtain torchvision's official ResNet18 IMAGENET1K_V1 initialization separately (https://download.pytorch.org/models/resnet18-f37072fd.pth), save as weights/resnet18-f37072fd.pth. The model verifies the f37072fd hash prefix and normalization. Nothing downloads implicitly. Model weights and data are not included or licensed by this repository.

## 2. Validate frozen source name partitions

```bash
python scripts/prepare_data.py --config configs/urban_to_rural.yaml --data-root data/LoveDA --output outputs/check_urban.json
python scripts/prepare_data.py --config configs/rural_to_urban.yaml --data-root data/LoveDA --output outputs/check_rural.json
```

## 3. Train each source model

```bash
python scripts/train.py --config configs/urban_to_rural.yaml --data-root data/LoveDA --weights weights/resnet18-f37072fd.pth --output runs/urban
python scripts/train.py --config configs/rural_to_urban.yaml --data-root data/LoveDA --weights weights/resnet18-f37072fd.pth --output runs/rural
```

Outputs include best.pt and history.json. Model selection uses only the frozen source selection names. Expected historical best epochs are18 and20, but equality on new hardware/software is not guaranteed. Do not overwrite the distributed result tables with a new run.

## 4. Infer source calibration and target caches

```bash
python scripts/infer.py --config configs/urban_to_rural.yaml --data-root data/LoveDA --weights weights/resnet18-f37072fd.pth --checkpoint runs/urban/best.pt --role calibration --output outputs/urban_source
python scripts/infer.py --config configs/urban_to_rural.yaml --data-root data/LoveDA --weights weights/resnet18-f37072fd.pth --checkpoint runs/urban/best.pt --role target --output outputs/rural_target
python scripts/infer.py --config configs/rural_to_urban.yaml --data-root data/LoveDA --weights weights/resnet18-f37072fd.pth --checkpoint runs/rural/best.pt --role calibration --output outputs/rural_source
python scripts/infer.py --config configs/rural_to_urban.yaml --data-root data/LoveDA --weights weights/resnet18-f37072fd.pth --checkpoint runs/rural/best.pt --role target --output outputs/urban_target
```

Each local NPZ contains `prob` float32 (7,H,W) and mapped `gt` (H,W). These are large local derived data, excluded by .gitignore. The manifest records source/target role and direction. Do not upload caches. `--role selection` is also available for diagnostics, never as the calibration negative pool.

## 5. Check source quantiles, evaluate frozen branches

```bash
python scripts/calibrate.py --cache outputs/urban_source --branch water_ratio --output outputs/water_quantile.json
python scripts/calibrate.py --cache outputs/rural_source --branch building_probability --output outputs/building_quantile.json
python scripts/evaluate.py --cache outputs/rural_target --branch water_ratio --output outputs/water_evaluation.json
python scripts/evaluate.py --cache outputs/urban_target --branch building_probability --output outputs/building_evaluation.json
```

Calibration refuses target-role manifests and uses all valid source negatives. It is a transparent exact in-RAM implementation, potentially requiring several GB for full calibration. It does not replace the large-data certified-tail optimization used historically. Returned quantiles can be compared with the frozen references; disagreement is a reproduction discrepancy, not permission to change historical thresholds. Evaluation deliberately uses the distributed frozen thresholds, not an automatically substituted newly fitted value. Exported JSON includes pooled baseline/rescue metrics and per-image B/H/W/M counts.

Run source ablations using `--branch water_probability` or `--branch building_ratio` and the corresponding source cache. Building ratio target evaluation is disabled because it was stopped historically. B2/C4 reproduction is available through the actual `loveda_rescue.spatial.spatial_prediction` API: supply probabilities, the matching local RGB array, the method name and `config['c4'][direction]`. B2/C4 are not recomputed from labels or GT; RGB is required for the historical C4 edge gate.

## 6. Source five-fold recalibration

```bash
python scripts/source_cv.py --cache outputs/urban_source --branch water_ratio --output outputs/cv_water_ratio.json
python scripts/source_cv.py --cache outputs/urban_source --branch water_probability --output outputs/cv_water_probability.json
python scripts/source_cv.py --cache outputs/rural_source --branch building_probability --output outputs/cv_building_probability.json
python scripts/source_cv.py --cache outputs/rural_source --branch building_ratio --output outputs/cv_building_ratio.json
```

Names are sorted, permuted with NumPy seed20261122 and partitioned into five image folds. Fit thresholds on four folds, evaluate the fifth. No training occurs. These random image folds are not independent geographic folds. Full recomputation requires local source probability caches and enough memory; the release only verifies this orchestration on tiny synthetic records.

## 7. Reproduce statistical figures

```bash
python scripts/reproduce_figures.py --output outputs/figures
```

All five figures use aggregate tables and histogram/bin counts bundled in results, with portable DejaVu Sans typography. Statistical redraw is verified separately from model reproduction. There are no RGB/GT examples or manuscript PDFs in the public bundle. Raw-layer spatial figures are intentionally excluded; users may compare their own locally generated predictions with local data, respecting the dataset terms. No claim is made that the original spatial case artwork is regenerated by this five-figure script.

## Verification boundary

Release checks execute imports, no-data tests, synthetic demo, all CLI help, a synthetic calibration/evaluation/five-fold route and all five statistical redraws. Full training, full-data inference, exact retraining reproducibility and original spatial case recreation are not executed or claimed during packaging. Scripts exclusively create their experiment output directories/files and fail on existing run outputs; choose a new output path for a new run.
