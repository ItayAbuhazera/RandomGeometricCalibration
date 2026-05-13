# Reproducing The IJCAI 2026 Paper

This document lists the paper artifact commands for **"Semantic Geometric Calibration in Randomized Neural Feature Space"**. Full reproduction is computationally expensive and requires trained checkpoints or retraining. CIFAR-10/100 are downloaded automatically; Tiny-ImageNet must be prepared manually.

## A. Environment Setup

Conda:

```bash
conda env create -f environment.yml
conda activate random-geometric-calibration
```

Pip-only alternative:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## B. Quick Smoke Test

```bash
python run_paper_experiments.py --quick
```

This runs CIFAR-10 / ResNet-18 / seed 11 and skips the Table 2-only GC(DAC) and GC(TULIP) variants.

## C. Full Table 1 Reproduction

```bash
python run_paper_experiments.py --full
python Experiments/generate_paper_tables.py --input-dir calibration_results
```

`--full` runs all 14 Table 1 dataset/model configurations across seeds 11-20.

## D. CIFAR-Only Convenience Run

```bash
python run_paper_experiments.py --cifar-full
```

This keeps the old CIFAR-only sweep behavior: 11 CIFAR dataset/model configurations across seeds 11-20.

## E. Tiny-ImageNet Dataset Note

Tiny-ImageNet is not downloaded by this repository. Extract it so the expected directory exists:

```text
data/tiny-imagenet-200/
  train/
  val/
  wnids.txt
  val/val_annotations.txt
```

The full paper run fails early with an actionable message if this directory does not contain the expected Tiny-ImageNet files.

## F. TOST Table

Table 2 uses paired TOST with δ = 0.5 pp. DINOv2 is included in Table 1 but excluded from the paired TOST configurations.

```bash
python Experiments/generate_paper_tables.py --input-dir calibration_results
```

## G. Hyperparameter Sensitivity

```bash
python Experiments/run_hyperparameter_sweep.py --skip-existing
python Experiments/generate_hyperparameter_table.py
```

The paper grid is `L in {2,4,6,8,10,12}` and `d in {64,128,256,512,1024}`.

## H. Beyond-ECE Metrics

```bash
python run_paper_experiments.py --full --emit-beyond-ece
python Experiments/generate_beyond_ece_table.py --input-dir calibration_results
```

Adaptive ECE is stored by default; MCE is emitted only when `--emit-beyond-ece` is used.

## I. Timing Figures

```bash
python Experiments/plot_end_to_end_timing.py --input-dir calibration_results
```

## J. Layer-Norm Figure

```bash
python Experiments/plot_layer_norms.py
```

## K. Output Layout

```text
results/models/{dataset}/{model}/seed{seed}/best_model.pth
calibration_results/{model}/{dataset}/baseline/{seed}/paper_results.json
```

## L. Computational Expectations

The full 14-configuration paper sweep across seeds 11-20 is expensive. A fresh clone can run the quick smoke test using the included quick-start checkpoint, but complete paper reproduction requires the remaining trained checkpoints or enough compute to retrain them.

Generated result files, tables, figures, logs, datasets, and large checkpoints are intentionally ignored by git. The tracked checkpoint exceptions are the small quick-start examples under `results/models/cifar10/resnet18/`.
