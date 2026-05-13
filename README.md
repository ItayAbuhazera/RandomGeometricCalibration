# Random Geometric Calibration (RGC)

Official code release for **"Semantic Geometric Calibration in Randomized Neural Feature Space"** (IJCAI 2026).

This repository contains the training and calibration pipeline for the Random Geometric Calibration experiments, plus a one-command wrapper for the available CIFAR sweep.

## Quick Start

```bash
conda env create -f environment.yml
conda activate random-geometric-calibration
pip install -r requirements.txt

# Quick smoke test: ResNet-18 / CIFAR-10 / seed 11, Table 1 methods only.
# GC(DAC) and GC(TULIP) are skipped in --quick (used only for Table 2 TOST tests).
python run_paper_experiments.py --quick

# Available full sweep (CIFAR configs only: 11 model-dataset pairs x 10 seeds = 110 runs)
python run_paper_experiments.py --full
```

A pre-trained checkpoint for the quick-start example is shipped at
`results/models/cifar10/resnet18/seed11/best_model.pth`, so the calibration
pipeline can be exercised without retraining. An additional seed-12 checkpoint
is also included for manual single-configuration checks.

## Where each paper artifact lives

| Paper artifact | Script | Notes |
|---|---|---|
| **Table 1** - ECE comparison (all methods, datasets, models) | [Experiments/run_rgc_experiments.py](Experiments/run_rgc_experiments.py) -> [Experiments/generate_paper_tables.py](Experiments/generate_paper_tables.py) | First produces per-config JSONs, then aggregates them into the LaTeX table. |
| **Table 2** - TOST equivalence tests (RGC_L vs GC(DAC), RGC_L vs RGC_C) | [Experiments/generate_paper_tables.py](Experiments/generate_paper_tables.py) | Uses the same per-config JSONs. |
| **Figure 1** - Hyperparameter sensitivity (L vs d) | [Experiments/run_hyperparameter_sweep.py](Experiments/run_hyperparameter_sweep.py) -> [Experiments/generate_hyperparameter_table.py](Experiments/generate_hyperparameter_table.py) | First produces one per-cell JSON for the L ∈ {2,4,6,8,10,12}, d ∈ {64,128,256,512,1024} sweep, then aggregates the grid to CSV/LaTeX. |
| **Figure 2** - Layer-wise feature norm comparison | [Experiments/plot_layer_norms.py](Experiments/plot_layer_norms.py) | Standalone; loads checkpoints and pools feature norms. |
| **Figure 3** - ECE normalized vs RGC_L | [Experiments/run_rgc_experiments.py](Experiments/run_rgc_experiments.py) | Per-config plot emitted alongside the JSON results. |
| **Figure 4** - Preprocessing time | [Experiments/plot_end_to_end_timing.py](Experiments/plot_end_to_end_timing.py) | Aggregates timing fields from the per-config JSONs. |
| **Figure 5** - Throughput | [Experiments/plot_end_to_end_timing.py](Experiments/plot_end_to_end_timing.py) | Same source data as Figure 4. |
| Backbone training | [Experiments/train_model.py](Experiments/train_model.py) | ResNet-18/50/101/152, DenseNet-121, DINOv2-Large. |
| "Beyond ECE" rankings (Adaptive ECE, MCE) | [Experiments/generate_beyond_ece_table.py](Experiments/generate_beyond_ece_table.py) | Opt-in; requires running the calibration pipeline with --emit-beyond-ece first. |

## Repository layout

```
run_paper_experiments.py        Top-level wrapper (training + calibration)

Experiments/
    train_model.py              Trains all paper backbones
    run_rgc_experiments.py      Per-config calibration pipeline (RGC_L, RGC_C, all baselines)
    run_hyperparameter_sweep.py Runs the Figure 1 L x d sensitivity sweep
    generate_hyperparameter_table.py Aggregates the Figure 1 sweep into CSV/LaTeX
    generate_paper_tables.py    Aggregates JSONs into Tables 1 and 2
    plot_layer_norms.py         Generates Figure 2
    plot_end_to_end_timing.py   Generates Figures 4 and 5

Calibrators/                    RGC, DAC, Temperature, Platt, Beta, Isotonic
Net/                            ResNet (CIFAR + Tiny-ImageNet variants), DenseNet, DINOv2
data/                           CIFAR-10/100, Tiny-ImageNet loaders
utils/                          Model loading, calibration utilities, geometry helpers
Metrics/                        ECE / accuracy
Losses/                         Loss factory used by training
results/, calibration_results/  Expected output paths (examples included)
```

## Architecture-agnostic layer discovery

The discovery/filtering procedure described in Section 3.1 of the paper is implemented in two places:
- `utils/layer_utils.py`: `filter_feature_layers`, `is_feature_layer` - the exclusion rules (activations, dropout, identity, flatten).
- `utils/coordinate_extraction.py`: `discover_coordinate_space` - walks `model.named_modules()`, runs a single forward pass to capture per-module output shapes, then filters with the rules above. The result is the `L_all` index set used by both RGCL and RGCC. For RGCL only, `utils/utils.py::discover_model_layers` provides the lighter-weight variant used during layer sampling.

## Direct entry-point examples

```bash
# Train a single backbone
python Experiments/train_model.py \
    --model-name resnet18 --dataset cifar10 --seed 12 \
    --output-root results/models

# Run RGC + all baselines on a trained backbone
python Experiments/run_rgc_experiments.py \
    --model-name resnet18 --dataset cifar10 \
    --training-method baseline --seed 12 \
    --results-base-dir results/models \
    --output-dir calibration_results

# Build the paper LaTeX tables from collected JSONs
python Experiments/generate_paper_tables.py --input-dir calibration_results

# Beyond-ECE rankings (Adaptive ECE + MCE), only if you ran the
# pipeline with --emit-beyond-ece
python run_paper_experiments.py --full --emit-beyond-ece
python Experiments/generate_beyond_ece_table.py --input-dir calibration_results

# Run the L x d hyperparameter sweep (Figure 1 data; expensive)
python Experiments/run_hyperparameter_sweep.py --skip-existing

# Aggregate sweep results into the sensitivity table
python Experiments/generate_hyperparameter_table.py

# Generate Figures 4 and 5
python Experiments/plot_end_to_end_timing.py --input-dir calibration_results

# Generate Figure 2
python Experiments/plot_layer_norms.py
```

## Output layout

```
results/models/{dataset}/{model}/seed{seed}/best_model.pth
calibration_results/{model}/{dataset}/baseline/{seed}/paper_results.json
```

## Datasets

- **CIFAR-10 / CIFAR-100** download automatically through `torchvision`.
- **Tiny-ImageNet** must be extracted into `data/tiny-imagenet-200/`.

## Installation notes

`requirements.txt` is the portable base install used for the paper workflow. FAISS is **not** required: the default path uses `fast_separation` with KNN by sklearn. Install FAISS only if you want to experiment with alternate nearest-neighbor backends:

```bash
conda install -c pytorch faiss-cpu     # or faiss-gpu on Linux
```

## Reproducing Figure 2

Figure 2 uses the torchvision pretrained DenseNet-121 to visualize feature norm growth in standard convnets; the custom CIFAR-trained DenseNet from `Net/densenet.py` is also supported via auto-detection.

## Citation

```bibtex
@inproceedings{abuhazera2026semantic,
  title     = {Semantic Geometric Calibration in Randomized Neural Feature Space},
  author    = {Itay Abuhazera and Liron Cohen and Gil Einziger},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence},
  year      = {2026},
  note      = {To appear}
}
```

