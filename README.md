# Random Geometric Calibration (RGC)

[![Paper](https://img.shields.io/badge/Paper-IJCAI--ECAI%202026-b31b1b.svg)](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2026/8155.pdf)
[![Conference](https://img.shields.io/badge/Conference-IJCAI--ECAI%202026-1f6feb.svg)](https://2026.ijcai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of [**“Semantic Geometric Calibration in Randomized Neural Feature Space”**](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2026/8155.pdf), accepted and presented at the **IJCAI–ECAI 2026 Main Track**.

**Authors:** Itay Abuhazera, Liron Cohen, and Gil Einziger

> **Paper status:** The linked PDF is the preliminary IJCAI–ECAI 2026 preprint. It will be replaced with the final proceedings version once it becomes available.

Random Geometric Calibration is a post-hoc confidence calibration method that builds randomized semantic feature spaces from neural representations. The release includes the two paper variants, `RGC_L` (randomized layer features) and `RGC_C` (randomized coordinate features), together with the baselines and table/figure generation scripts used in the paper.

## Installation

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

## Quick Smoke Test

```bash
python run_paper_experiments.py --quick
```

The quick path uses CIFAR-10 / ResNet-18 / seed 11. A small release checkpoint is included at `results/models/cifar10/resnet18/seed11/best_model.pth`.

## Reproducing the Paper

See [REPRODUCING.md](REPRODUCING.md) for the exact commands for Table 1, Table 2 TOST, hyperparameter sensitivity, beyond-ECE metrics, timing figures, and layer-norm figures.

The main entry points are:

```bash
python run_paper_experiments.py --full
python Experiments/generate_paper_tables.py --input-dir calibration_results
```

`--full` runs the 14 dataset/model configurations used for Table 1. For a CIFAR-only convenience run, use:

```bash
python run_paper_experiments.py --cifar-full
```

## Repository Layout

```text
run_paper_experiments.py        Top-level paper experiment runner
Experiments/                    Training, calibration, sweeps, and table/figure scripts
Calibrators/                    RGC and calibration baselines
Net/                            CIFAR and Tiny-ImageNet model definitions
data/                           Dataset loaders
utils/                          Model loading and geometric feature utilities
Metrics/                        ECE and related calibration metrics
paper_manifest.json             Reproducibility contract for the IJCAI-ECAI 2026 paper
```

## Datasets

CIFAR-10 and CIFAR-100 are downloaded automatically by `torchvision`.

Tiny-ImageNet must be downloaded manually and extracted into:

```text
data/tiny-imagenet-200/
```

See [data/tiny-imagenet-200/README.md](data/tiny-imagenet-200/README.md) for the expected layout.

## Citation

If you use this code or build on Random Geometric Calibration, please cite the paper:

```bibtex
@inproceedings{abuhazera2026semantic,
  title     = {Semantic Geometric Calibration in Randomized Neural Feature Space},
  author    = {Abuhazera, Itay and Cohen, Liron and Einziger, Gil},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year      = {2026},
  note      = {To appear}
}
```

The citation will be updated with the official proceedings URL, page numbers, and DOI once they become available.

## License

This project is released under the [MIT License](LICENSE).
