#!/usr/bin/env python3
"""
Train classification models on CIFAR-10, CIFAR-100, or Tiny ImageNet.

Supported models:
    - ResNet-18, ResNet-50, ResNet-101, ResNet-152
    - DenseNet-121
    - DINOv2 variants (with frozen backbone)

Outputs (per seed):
    results/models/
        {dataset}/{model_name}/seed{seed}/
            best_model.pth
            config.json
            results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

# Ensure the repository root is on sys.path so we can import project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.calibration_utils import compute_ece
from Losses.loss import get_loss_function
from Net.resnet_cifar import resnet18, resnet50, resnet101, resnet152
from Net.densenet import densenet121
from Net.resnet_tiny_imagenet import resnet18 as tiny_resnet18
from Net.resnet_tiny_imagenet import resnet50 as tiny_resnet50
from Net.resnet_tiny_imagenet import resnet101 as tiny_resnet101
from Net.resnet_tiny_imagenet import resnet152 as tiny_resnet152
from data.cifar10 import get_train_valid_loader as cifar10_get_train_valid_loader
from data.cifar100 import get_train_valid_loader as cifar100_get_train_valid_loader
from data.tiny_imagenet import (
    get_data_loader as tiny_imagenet_get_data_loader,
    TinyImageNet,
)

# Force unbuffered output
(
    sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stdout, "reconfigure")
    else None
)
from utils.logging_config import setup_logging, get_logger

setup_logging()

logger = get_logger(__name__)


class DINOv2Classifier(nn.Module):
    """
    DINOv2 pretrained backbone with trainable linear classifier.
    Backbone is frozen; only the head is trained.
    """

    def __init__(
        self,
        model_name: str = "dinov2_large",
        num_classes: int = 100,
        freeze_backbone: bool = True,
        input_size: int = 32,
    ):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size

        # Map model names to timm model strings
        timm_name_map = {
            "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
            "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
            "dinov2_large": "vit_large_patch14_dinov2.lvd142m",
            "dinov2_giant": "vit_giant_patch14_dinov2.lvd142m",
        }

        timm_name = timm_name_map.get(model_name.lower())
        if timm_name is None:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. Choose from {list(timm_name_map.keys())}"
            )

        # Load pretrained backbone
        self.backbone = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=224,  # DINOv2 expects 224x224
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Trainable classifier head
        self.head = nn.Linear(feat_dim, num_classes)

        # Resize transform for small inputs (CIFAR)
        self.needs_resize = input_size < 224
        if self.needs_resize:
            self.resize = nn.Upsample(
                size=(224, 224), mode="bilinear", align_corners=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_resize:
            x = self.resize(x)

        # Extract features from backbone (backbone is frozen but we need gradients for head)
        # Even though backbone params are frozen, we need to allow gradients to flow
        # through the computation graph for the head to train
        features = self.backbone(x)

        return self.head(features)

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen."""
        super().train(mode)
        # Always keep backbone in eval mode if parameters are frozen
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self


@dataclass
class TrainingConfig:
    method: str = "baseline_cross_entropy"
    dataset: str = "cifar10"
    model: str = "resnet18"
    seed: int = 11
    num_classes: int = 10
    epochs: int = 350
    batch_size: int = 64
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-3
    warmup_epochs: int = 10
    accuracy_threshold: float = 0.96
    temperature_init: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classification models on CIFAR-10/100 or Tiny ImageNet."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cifar-10-batches-py"),
        help="Path to CIFAR-10 data directory (or its parent).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "models"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--experiment-group", type=str, default="baseline")
    parser.add_argument("--method", type=str, default="baseline_cross_entropy")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--target-acc", type=float, default=96.0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--temperature-init", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=20,
        help="Number of epochs with no improvement after which learning rate will be reduced (for ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.2,
        help="Factor by which learning rate will be reduced (for ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate (for ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=40,
        help="Number of epochs with no improvement after LR decay before stopping training. Set to 0 to disable early stopping.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP) training.",
    )
    return parser.parse_args()


def resolve_cifar_root(path_str: str) -> Path:
    """
    Torchvision expects the *parent* directory that will contain the CIFAR
    data folder (e.g. ``cifar-10-batches-py`` or ``cifar-100-python``).

    If the user passes the inner directory directly, strip the final
    component automatically so that both of these work:

        --data-root data
        --data-root data/cifar-10-batches-py
        --data-root data/cifar-100-python
    """
    path = Path(path_str).expanduser().resolve()
    if path.name in {"cifar-10-batches-py", "cifar-100-python"}:
        return path.parent
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(
    seed: int,
    batch_size: int,
    pin_memory: bool,
    dataset_name: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/validation dataloaders for CIFAR-10, CIFAR-100, or Tiny ImageNet.

    Uses the same data loaders from data/cifar10.py and data/cifar100.py
    to ensure consistency with the calibration pipeline in
    Experiments/run_rgc_experiments.py.

    The dataset is selected via ``dataset_name``:
      - ``"cifar100"`` -> Data/cifar100.py loader
      - ``"tiny_imagenet"`` -> Data/tiny_imagenet.py loader
      - anything else  -> Data/cifar10.py loader (default)
    """
    ds_name = (dataset_name or "cifar10").lower()

    if ds_name == "cifar100":
        train_loader, val_loader = cifar100_get_train_valid_loader(
            batch_size=batch_size,
            augment=True,  # Training augmentation
            random_seed=seed,
            valid_size=0.1,
            pin_memory=pin_memory,
        )
    elif ds_name == "tiny_imagenet":
        from data.tiny_imagenet import (
            get_train_valid_loader as tiny_imagenet_get_train_valid_loader,
        )

        train_loader, val_loader = tiny_imagenet_get_train_valid_loader(
            batch_size=batch_size,
            augment=True,
            random_seed=seed,
            valid_size=0.1,
            pin_memory=pin_memory,
        )
    else:
        train_loader, val_loader = cifar10_get_train_valid_loader(
            batch_size=batch_size,
            augment=True,  # Training augmentation
            random_seed=seed,
            valid_size=0.1,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


def create_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Use lower LR for DINOv2 linear head
    if args.model_name.startswith("dinov2"):
        lr = args.lr * 0.1  # Lower LR for linear probing
    else:
        lr = args.lr

    optimizer = optim.SGD(
        trainable_params,  # Changed from model.parameters()
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # Use ReduceLROnPlateau for ResNet models; cosine warmup for DenseNet / DINOv2.
    if args.model_name in {"resnet18", "resnet50", "resnet101", "resnet152"}:
        # Use ReduceLROnPlateau for adaptive LR scheduling based on validation performance
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # 'max' because we're monitoring validation accuracy (higher is better)
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )
    else:
        cosine_epochs = max(1, args.epochs - args.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        if args.warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=args.warmup_epochs,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = cosine
    return optimizer, scheduler


def training_paths(args: argparse.Namespace) -> Dict[str, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    seed_dir = (
        output_root
        / args.dataset
        / args.model_name
        / f"seed{args.seed}"
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    return {
        "seed_dir": seed_dir,
        "exp_dir": seed_dir,
        "config": seed_dir / "config.json",
        "results": seed_dir / "results.json",
        "best_model": seed_dir / "best_model.pth",
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
    ece_bins: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, targets) / targets.size(0)
            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += targets.size(0)
            probs_all.append(torch.softmax(logits, dim=1).cpu())
            labels_all.append(targets.cpu())

    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    probs_np = torch.cat(probs_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    ece = compute_ece(probs_np, labels_np, n_bins=ece_bins)
    return {"loss": float(avg_loss), "accuracy": float(acc), "ece": float(ece)}


def run_training(args: argparse.Namespace) -> None:
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    set_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader, val_loader = build_dataloaders(
        seed=args.seed,
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        dataset_name=args.dataset,
    )

    # Model selection: ResNet18/50/101/152, DenseNet121, DINOv2 variants
    # Determine if we should use Tiny ImageNet-specific models
    use_tiny_imagenet_models = args.dataset.lower() == "tiny_imagenet"

    if args.model_name == "resnet18":
        if use_tiny_imagenet_models:
            model = tiny_resnet18(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
        else:
            model = resnet18(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
    elif args.model_name == "resnet50":
        if use_tiny_imagenet_models:
            model = tiny_resnet50(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
        else:
            model = resnet50(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
    elif args.model_name == "densenet121":
        model = densenet121(
            num_classes=args.num_classes,
            temp=args.temperature_init,
        ).to(device)
    elif args.model_name == "resnet101":
        if use_tiny_imagenet_models:
            model = tiny_resnet101(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
        else:
            model = resnet101(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
    elif args.model_name == "resnet152":
        if use_tiny_imagenet_models:
            model = tiny_resnet152(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
        else:
            model = resnet152(
                num_classes=args.num_classes,
                temp=args.temperature_init,
            ).to(device)
    elif args.model_name.startswith("dinov2"):
        # Determine input size based on dataset
        if args.dataset.lower() in ["cifar10", "cifar100"]:
            input_size = 32
        elif args.dataset.lower() == "tiny_imagenet":
            input_size = 64
        else:
            input_size = 224

        model = DINOv2Classifier(
            model_name=args.model_name,
            num_classes=args.num_classes,
            freeze_backbone=True,  # Freeze backbone, train only head
            input_size=input_size,
        ).to(device)

        logger.info(f"Created DINOv2 model: {args.model_name}")
        logger.info(f"  Backbone frozen: True")
        logger.info(f"  Input resize: {input_size} -> 224")
        logger.info(
            f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    scaler = torch.amp.GradScaler(device.type, enabled=not args.no_amp and device.type == "cuda")
    loss_fn = get_loss_function("cross_entropy")

    paths = training_paths(args)
    config = TrainingConfig(
        method=args.method,
        dataset=args.dataset,
        model=args.model_name,
        seed=args.seed,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        accuracy_threshold=args.target_acc / 100.0,
        temperature_init=args.temperature_init,
    )
    with open(paths["config"], "w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, indent=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_ece": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    target_acc = args.target_acc

    # Early stopping tracking
    epochs_no_improve = 0
    epochs_since_lr_decay = 0
    last_lr = optimizer.param_groups[0]["lr"]
    lr_has_decayed = False  # Track if LR has decayed at least once
    use_reduce_lr_on_plateau = isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau
    )

    # Flags to track when early stopping conditions are met (for exit at epoch 100)
    should_stop_target_acc = False
    should_stop_patience = False

    # Adjust target accuracy for dataset difficulty
    if args.dataset.lower() == "tiny_imagenet" and target_acc > 70.0:
        logger.warning(
            f"Target accuracy {target_acc}% is unrealistic for Tiny ImageNet. "
            f"State-of-the-art is ~65%. Consider lowering --target-acc or training will run all epochs."
        )

    if use_reduce_lr_on_plateau:
        logger.info(
            f"Using ReduceLROnPlateau scheduler: "
            f"patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.lr_min}"
        )
        if args.early_stop_patience > 0:
            logger.info(
                f"Early stopping enabled: will stop after {args.early_stop_patience} epochs "
                f"with no improvement following LR decay"
            )
    elif args.early_stop_patience > 0:
        logger.info(
            f"Early stopping patience ({args.early_stop_patience}) is set. "
            f"Using simple patience-based early stopping (tracks epochs without improvement, "
            f"works with any scheduler including {type(scheduler).__name__})."
        )

    start_time = time.time()
    total_train_samples = len(train_loader.dataset)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        seen = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                logits = model(images)
                loss = loss_fn(logits, targets) / targets.size(0)

            # Skip batch if loss is NaN/Inf (FP16 overflow protection)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            # Unscale and check for NaN gradients before clipping/stepping
            scaler.unscale_(optimizer)

            # Check for NaN gradients - skip step if any found
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            seen += targets.size(0)

        train_loss_avg = epoch_loss / max(1, seen)
        train_acc = 100.0 * correct / max(1, seen)

        # Progress log every 10 epochs (train-only summary).
        if epoch % 10 == 0:
            logger.info(
                f"[Train {epoch:03d}] "
                f"Loss {train_loss_avg:.4f} | Acc {train_acc:.2f}%"
            )

        val_metrics = evaluate_model(model, val_loader, device, loss_fn, args.ece_bins)

        history["train_loss"].append(float(train_loss_avg))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_ece"].append(val_metrics["ece"])

        # Check for improvement
        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_no_improve = 0
            # Reset counters and flags on improvement
            if args.early_stop_patience > 0:
                if use_reduce_lr_on_plateau and lr_has_decayed:
                    epochs_since_lr_decay = 0
                should_stop_patience = False  # Reset flag since we have improvement
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "args": vars(args),
                },
                paths["best_model"],
            )
        else:
            epochs_no_improve += 1
            # Increment counter only if LR has decayed and we're tracking for early stopping
            if lr_has_decayed and args.early_stop_patience > 0:
                epochs_since_lr_decay += 1
                if epoch % 10 == 0:  # Log every 10 epochs for debugging
                    logger.debug(
                        f"Epoch {epoch}: No improvement. Counter incremented to {epochs_since_lr_decay} "
                        f"(lr_has_decayed={lr_has_decayed})"
                    )

        # Update learning rate scheduler
        # ReduceLROnPlateau needs the metric, others use step() without arguments
        current_lr = optimizer.param_groups[0]["lr"]
        if use_reduce_lr_on_plateau:
            # Log counter value before LR scheduler step for debugging
            if args.early_stop_patience > 0 and lr_has_decayed:
                logger.debug(
                    f"Before LR scheduler step (epoch {epoch}): "
                    f"counter={epochs_since_lr_decay}, lr_has_decayed={lr_has_decayed}"
                )
            scheduler.step(val_metrics["accuracy"])
            # Check if LR was reduced
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                logger.info(
                    f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f} "
                    f"at epoch {epoch} (no improvement for {args.lr_patience} epochs)"
                )
                # Mark that LR has decayed at least once (don't reset counter on subsequent decays)
                if not lr_has_decayed:
                    lr_has_decayed = True
                    epochs_since_lr_decay = (
                        0  # Only reset on first decay to start tracking
                    )
                    logger.info(
                        f"First LR decay detected at epoch {epoch}. Starting early stopping counter."
                    )
                else:
                    logger.info(
                        f"Subsequent LR decay at epoch {epoch}. "
                        f"Counter NOT reset (continuing from: {epochs_since_lr_decay})"
                    )
                # Note: We do NOT reset epochs_since_lr_decay on subsequent LR decays.
                # The counter should only reset on improvement, allowing us to track
                # total epochs without improvement since the first LR decay.
                last_lr = new_lr
        else:
            scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            lr_info = f"LR {optimizer.param_groups[0]['lr']:.6f}"
            if use_reduce_lr_on_plateau and args.early_stop_patience > 0:
                lr_info += f" | Epochs since LR decay: {epochs_since_lr_decay}/{args.early_stop_patience}"
            logger.info(
                f"[Epoch {epoch:03d}] "
                f"Train Loss {train_loss_avg:.4f} | Train Acc {train_acc:.2f}% | "
                f"Val Acc {val_metrics['accuracy']:.2f}% | Val ECE {val_metrics['ece']:.4f} | "
                f"Best {best_val_acc:.2f}% (Epoch {best_epoch}) | {lr_info}"
            )

        # Hard limit: stop at 200 epochs
        if epoch >= 200:
            logger.info(
                f"Reached maximum epoch limit (200). "
                f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}. "
                "Stopping training."
            )
            break

        # Check target accuracy condition (check every epoch)
        if best_val_acc >= target_acc:
            should_stop_target_acc = True
            if epoch >= 100:
                logger.info(
                    f"Target accuracy {target_acc:.2f}% reached at epoch {best_epoch}. "
                    "Stopping early."
                )
                break
            else:
                logger.info(
                    f"Target accuracy {target_acc:.2f}% reached at epoch {best_epoch}. "
                    f"Will exit at epoch 100 (current: {epoch})."
                )

        # Check patience-based early stopping condition (check every epoch)
        if args.early_stop_patience > 0:
            if use_reduce_lr_on_plateau and lr_has_decayed:
                # For ReduceLROnPlateau: track epochs since first LR decay
                if epochs_since_lr_decay >= args.early_stop_patience:
                    should_stop_patience = True
                    if epoch >= 100:
                        logger.info(
                            f"No improvement for {args.early_stop_patience} epochs after first LR decay. "
                            f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}. "
                            f"Total epochs without improvement since first LR decay: {epochs_since_lr_decay}. "
                            "Stopping training."
                        )
                        break
                    else:
                        logger.info(
                            f"Early stopping condition met ({epochs_since_lr_decay} epochs without improvement). "
                            f"Will exit at epoch 100 (current: {epoch})."
                        )
            else:
                # For other schedulers (e.g., cosine annealing): simple patience-based stopping
                # Track epochs without improvement (independent of LR changes)
                if epochs_no_improve >= args.early_stop_patience:
                    should_stop_patience = True
                    if epoch >= 100:
                        logger.info(
                            f"No improvement for {args.early_stop_patience} epochs. "
                            f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}. "
                            f"Total epochs without improvement: {epochs_no_improve}. "
                            "Stopping training."
                        )
                        break
                    else:
                        logger.info(
                            f"Early stopping condition met ({epochs_no_improve} epochs without improvement). "
                            f"Will exit at epoch 100 (current: {epoch})."
                        )

        # Exit at epoch 100 if conditions were met earlier
        if epoch == 100:
            if should_stop_target_acc:
                logger.info(
                    f"Reached epoch 100. Target accuracy condition was met at epoch {best_epoch}. "
                    f"Best validation accuracy: {best_val_acc:.2f}%. Stopping training."
                )
                break
            elif should_stop_patience:
                if use_reduce_lr_on_plateau:
                    logger.info(
                        f"Reached epoch 100. Patience-based early stopping condition was met. "
                        f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}. "
                        f"Total epochs without improvement since first LR decay: {epochs_since_lr_decay}. "
                        "Stopping training."
                    )
                else:
                    logger.info(
                        f"Reached epoch 100. Patience-based early stopping condition was met. "
                        f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}. "
                        f"Total epochs without improvement: {epochs_no_improve}. "
                        "Stopping training."
                    )
                break

    training_time = time.time() - start_time

    # Reload the best checkpoint before final evaluation.
    checkpoint = torch.load(paths["best_model"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = evaluate_model(model, val_loader, device, loss_fn, args.ece_bins)

    results = {
        "experiment_name": f"{args.method}_{args.dataset}_{args.model_name}_seed{args.seed}",
        "config": asdict(config),
        "training_time": training_time,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "training_history": history,
        "evaluation_results": final_metrics,
    }

    with open(paths["results"], "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info(f"Finished. Best model saved to: {paths['best_model']}")
    logger.info(f"   Metrics JSON: {paths['results']}")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
