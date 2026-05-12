# Random Geometric Calibration (RGC)

Official code release for **"Semantic Geometric Calibration in Randomized Neural Feature Space"** (IJCAI 2026).

This repository contains the complete training and calibration pipeline used to produce every table and figure in the paper, plus a one-command wrapper for reproducing the headline results.

## Quick Start

```bash
conda env create -f environment.yml
conda activate geometric-internal-calibration
pip install -r requirements.txt

# Quick smoke test: ResNet-18 / CIFAR-10 / seed 11, Table 1 methods only.
# GC(DAC) and GC(TULIP) are skipped in --quick (used only for Table 2 TOST tests).
python run_paper_experiments.py --quick

# Full paper reproduction (14 model-dataset pairs x 3 seeds = 42 runs)
python run_paper_experiments.py --full
```

A pre-trained checkpoint is shipped at
`results/models/cifar10/resnet18/seed12/best_model.pth`
so the calibration pipeline can be exercised without retraining.

## Where each paper artifact lives

| Paper artifact | Script | Notes |
|---|---|---|
| **Table 1** - ECE comparison (all methods, datasets, models) | [Experiments/run_rgc_experiments.py](Experiments/run_rgc_experiments.py) -> [Experiments/generate_paper_tables.py](Experiments/generate_paper_tables.py) | First produces per-config JSONs, then aggregates them into the LaTeX table. |
| **Table 2** - TOST equivalence tests (RGC_L vs GC(DAC), RGC_L vs RGC_C) | [Experiments/generate_paper_tables.py](Experiments/generate_paper_tables.py) | Uses the same per-config JSONs. |
| **Figure 1** - Hyperparameter sensitivity (L vs d) | Generated from the hyperparameter sweep in [calibration_results/](calibration_results/) | Currently produced by ad-hoc analysis of the per-config JSON files rather than a dedicated script. |
| **Figure 2** - Layer-wise feature norm comparison | [Experiments/plot_layer_norms.py](Experiments/plot_layer_norms.py) | Standalone; loads checkpoints and pools feature norms. |
| **Figure 3** - ECE normalized vs RGC_L | [Experiments/run_rgc_experiments.py](Experiments/run_rgc_experiments.py) | Per-config plot emitted alongside the JSON results. |
| **Figure 4** - Preprocessing time | [Experiments/plot_end_to_end_timing.py](Experiments/plot_end_to_end_timing.py) | Aggregates timing fields from the per-config JSONs. |
| **Figure 5** - Throughput | [Experiments/plot_end_to_end_timing.py](Experiments/plot_end_to_end_timing.py) | Same source data as Figure 4. |
| Backbone training | [Experiments/train_model.py](Experiments/train_model.py) | ResNet-18/50/101/152, DenseNet-121, DINOv2-Large. |

## Repository layout

```
run_paper_experiments.py        Top-level wrapper (training + calibration)

Experiments/
    train_model.py              Trains all paper backbones
    run_rgc_experiments.py      Per-config calibration pipeline (RGC_L, RGC_C, all baselines)
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
@inproceedings{rgc2026,
  title={Semantic Geometric Calibration in Randomized Neural Feature Space},
  author={Itay Abuhazera, Liron Cohen, Gil Einziger},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```
