# Hass Avocado Ripening — Paper Reproduction (Foods 2024, 13, 1150)

**5-stage ripeness classification** of avocados from a single RGB photo + **per-storage-group shelf-life estimation**.
Reproduces the published method of Xavier et al. (Foods 2024, DOI 10.3390/foods13081150) as faithfully as possible.
Data: Mendeley DOI 10.17632/3xd9n945v8.1 (CC BY 4.0).

> ⚠️ The model input is **a single RGB image only**. Metadata such as Storage Group, Day, Sample, view, days_left, and color statistics
> are not fed to the model (CLAUDE.md §2.3). Storage group / Sample / view / Day are used only for
> filtering, sample-level split, two-side aggregation, choosing the shelf-life formula, and analysis.

## Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy scikit-learn pillow pyyaml openpyxl matplotlib
```
Data: `data/Avocado Ripening Dataset.xlsx` (committed) + download the images via DOI to
`data/Hass Avocado Ripening Photographic Dataset/.../Avocado Ripening Dataset/*.jpg`.

## Reproduction steps (paper reproduction §19)

```bash
# 1) Data integrity checks → metadata_clean.csv (12-item check)
python -m src.validate_data                 # (--skip-image-check to skip the corruption check)

# 2) Sample-level 70/15/15 split → splits.csv (shared by all models, leakage-verification assert)
python -m src.split --seed 42

# 3) Pipeline check (smoke)
python -m src.train --config configs/paper/general_resnet18.yaml --smoke-test

# 4) Individual training (e.g., General × ResNet-18). --val-freq is a practical override
python -m src.train --config configs/paper/general_resnet18.yaml --val-freq 80

# 5) Train + evaluate all 8 experiments
python scripts/run_paper_experiments.py

# 6) Evaluation only (against trained checkpoints)
python -m src.evaluate --all
python -m src.evaluate --config configs/paper/general_resnet18.yaml

# 7) Pipeline tests (§17, 11 assertions)
python tests/test_pipeline.py
```

## 8 experiments (5-stage)
`configs/paper/*.yaml` — {general, T10, T20, Tamb} × {ResNet-18, AlexNet}.
experiment_id: `P1_general_resnet18_...` through `P8_tamb_alexnet_...`.

## Output file structure
```
outputs/paper_reproduction/
├── splits/         metadata_clean.csv, splits.csv
├── checkpoints/    <experiment_id>/best.pt, config.json, history.json
├── predictions/    <experiment_id>_test_predictions.csv, _test_observations.csv
├── metrics/        <experiment_id>_metrics.json  (A image / B best-side / C two-side / shelf-life)
├── confusion_matrices/  <experiment_id>_confusion.json
├── logs/
└── reports/        data_validation.json, paper_reproduction_report.md
```
Checkpoints (*.pt) and images are excluded from commits via .gitignore.

## Code structure (src/)
`data.py` (metadata loading) · `validate_data.py` (integrity) · `split.py` (sample-level split) ·
`transforms.py` (paper/eval/noaug) · `sampler.py` (random oversampling) · `dataset.py` (image Dataset) ·
`models.py` (ResNet-18/AlexNet) · `train.py` (config-driven training) · `evaluate.py` (image/best-side/two-side) ·
`shelf_life.py` (per-storage-group α formula) · `metrics.py` · `utils.py`.

## Preserving the existing baseline (§14)
The no-augmentation General ResNet-18 baseline (**B0**, single-image val acc 0.783) must not be deleted or overwritten.
Preserve the checkpoint `checkpoints/resnet18_baseline.pt`; results are in `docs/modeling-plan.md`.

For detailed reproduced items / undisclosed items / results, see **[`docs/paper_reproduction_report.md`](docs/paper_reproduction_report.md)**
(running the pipeline also regenerates an updated copy under `outputs/paper_reproduction/reports/`).
