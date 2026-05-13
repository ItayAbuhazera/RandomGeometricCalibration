# Random Geometric Calibration (RGC)

Official code for **"Semantic Geometric Calibration in Randomized Neural Feature Space"** (IJCAI 2026). Random Geometric Calibration is a post-hoc confidence calibration method that builds randomized semantic feature spaces from neural representations. The release includes the two paper variants, RGC_L (randomized layer features) and RGC_C (randomized coordinate features), together with the baselines and table/figure generation scripts used in the paper.

## Installation

```bash
conda env create -f environment.yml
conda activate random-geometric-calibration
pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python run_paper_experiments.py --quick
```

The quick path uses CIFAR-10 / ResNet-18 / seed 11. A small release checkpoint is included at `results/models/cifar10/resnet18/seed11/best_model.pth`.

## Reproducing The Paper

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
paper_manifest.json             Reproducibility contract for the IJCAI 2026 paper
```

## Datasets

CIFAR-10 and CIFAR-100 are downloaded automatically by `torchvision`.

Tiny-ImageNet must be downloaded manually and extracted into:

```text
data/tiny-imagenet-200/
```

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

## License

This project is released under the [MIT License](LICENSE).
