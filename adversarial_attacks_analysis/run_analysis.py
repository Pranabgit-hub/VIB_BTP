import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =========================================================
# PATHS
# =========================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ANALYSIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    ANALYSIS_DIR
)


# =========================================================
# IMPORTS
# =========================================================

from model import build_vib

from divergences import divergence_loss

from entropy import (
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy
)

from information_metrics import (
    calculate_all_information_metrics
)

from adversarial_attacks import (
    fgsm_attack,
    pgd_attack,
    evaluate_under_attack,
    generate_adversarial_batch
)


# =========================================================
# CONFIGURATION
#
# Global defaults below apply to every dataset. Individual
# datasets may override them via optional keys in
# DATASET_REGISTRY (see "epochs" / "hidden_dim" /
# "latent_dim" / "epsilons" / "batch_size" overrides).
# =========================================================

BATCH_SIZE = 128

EPOCHS = 20

# ---------------------------------------------------------
# Compute-load control
#
# Fractions of the train/test splits actually used.
#
# 1.0 = full dataset (original behaviour).
# Values < 1.0 subsample reproducibly (seeded) so the run
# finishes on CPU within practical time limits.
# Set back to 1.0 for final full-data experiments.
# ---------------------------------------------------------

TRAIN_SUBSET_FRACTION = 1.0

TEST_SUBSET_FRACTION = 1.0

LEARNING_RATE = 1e-3

BETA = 1e-5

LATENT_DIM = 32

HIDDEN_DIM = 256

# ---------------------------------------------------------
# Dataset switch + registry
#
# To ADD A NEW DATASET: add one entry to DATASET_REGISTRY
# and set DATASET_NAME above. Nothing else changes.
#
# Required keys per entry:
#   dataset_class    torchvision dataset class
#   input_shape      (C, H, W)
#   flatten          True -> batches flattened to (B, C*H*W)
#                    for the MLP; False -> image tensors for CNN
#   arch             "mlp" or "cnn" (key of ARCH_REGISTRY)
#   num_classes      number of output classes
#   train_transform  augmentation/preprocessing (train split)
#   test_transform   preprocessing (test split — keep clean,
#                    no augmentation, so adversarial eval is fair)
#
# Optional keys (override global defaults):
#   epochs, batch_size, hidden_dim, latent_dim, epsilons,
#   train_subset_fraction, test_subset_fraction
# ---------------------------------------------------------

DATASET_NAME = "cifar10"

DATASET_REGISTRY = {

    "mnist": {
        "dataset_class": datasets.MNIST,
        "input_shape": (1, 28, 28),
        "flatten": True,
        "arch": "mlp",
        "num_classes": 10,
        "train_transform": transforms.ToTensor(),
        "test_transform": transforms.ToTensor(),
    },

    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "input_shape": (3, 32, 32),
        "flatten": False,
        "arch": "cnn",
        "num_classes": 10,
        "train_transform": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        "test_transform": transforms.ToTensor(),
        # ---- CIFAR-10 specific tuning ----
        #
        # CPU-friendly balanced load:
        #   15 epochs on a reproducible 40% train subset
        #   (20k images), batch 256 -> ~79 batches/epoch.
        #   Adversarial eval uses a 50% test subset (5k)
        #   and PGD steps are halved globally below.
        # Roughly an order of magnitude faster than the
        # original 50-epoch full-data configuration while
        # preserving the qualitative comparison.
        "epochs": 15,
        "batch_size": 256,
        "hidden_dim": 256,
        "latent_dim": 64,
        "train_subset_fraction": 0.4,
        "test_subset_fraction": 0.5,
        "epsilons": [0.02, 0.031, 0.05, 0.08, 0.12],
    },
}

if DATASET_NAME not in DATASET_REGISTRY:
    raise ValueError(
        f"Unknown DATASET_NAME: {DATASET_NAME}. "
        f"Choose from: {list(DATASET_REGISTRY.keys())}"
    )

DATASET_CONFIG = DATASET_REGISTRY[DATASET_NAME]

INPUT_SHAPE = DATASET_CONFIG["input_shape"]

FLATTEN_INPUTS = DATASET_CONFIG["flatten"]

ARCHITECTURE = DATASET_CONFIG["arch"]

NUM_CLASSES = DATASET_CONFIG["num_classes"]

INPUT_DIM = 1

for _dim in INPUT_SHAPE:

    INPUT_DIM *= _dim


# Per-dataset hyperparameter overrides

EPOCHS = DATASET_CONFIG.get("epochs", EPOCHS)

BATCH_SIZE = DATASET_CONFIG.get("batch_size", BATCH_SIZE)

TRAIN_SUBSET_FRACTION = DATASET_CONFIG.get(
    "train_subset_fraction",
    TRAIN_SUBSET_FRACTION
)

TEST_SUBSET_FRACTION = DATASET_CONFIG.get(
    "test_subset_fraction",
    TEST_SUBSET_FRACTION
)

HIDDEN_DIM = DATASET_CONFIG.get("hidden_dim", HIDDEN_DIM)

LATENT_DIM = DATASET_CONFIG.get("latent_dim", LATENT_DIM)


TRAIN_TRANSFORM = DATASET_CONFIG["train_transform"]

TEST_TRANSFORM = DATASET_CONFIG["test_transform"]


# ---------------------------------------------------------
# Alpha values — matching V3
# ---------------------------------------------------------

RENYI_ALPHA = 0.95

TSALLIS_ALPHA = 0.95


SEED = 42


DIVERGENCES = [
    "kl",
    "renyi",
    "tsallis"
]


# ---------------------------------------------------------
# Adversarial attack configuration
#
# Datasets may override the epsilon list via the optional
# "epsilons" registry key (CIFAR-10 uses a smaller range:
# standard CIFAR-10 PGD evaluation is eps=8/255 ~= 0.031).
# ---------------------------------------------------------

DEFAULT_EPSILONS = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.3
]

EPSILONS = DATASET_CONFIG.get(
    "epsilons",
    DEFAULT_EPSILONS
)

PGD_STEPS = 10

PGD_STEP_SIZE = None  # defaults to epsilon / 4


# ---------------------------------------------------------
# ADVERSARIAL TRAINING CONFIGURATION
#
# FGSM / PGD appear in TWO distinct roles in this pipeline:
#
#   1. TRAINING  (adversarial training, Madry-style):
#      each training batch is replaced by an adversarial
#      version crafted on-the-fly with the chosen method,
#      and the VIB objective is minimized on those examples.
#      Controlled by TRAINING_MODES below.
#
#   2. EVALUATION (unchanged): after training, every model —
#      clean-trained or adversarially-trained — is probed
#      with white-box FGSM and PGD attacks at all EPSILONS.
#
# TRAINING_MODES selects which training variants run:
#   "clean" -> standard training (original behaviour)
#   "fgsm"  -> FGSM-adversarial training
#   "pgd"   -> PGD-adversarial training (num_steps =
#              ADV_TRAIN_PGD_STEPS)
# Each mode writes its results to its own subfolder under
# results/<dataset>/ so all variants stay comparable.
# ---------------------------------------------------------

TRAINING_MODES = [
    "clean",
    "fgsm",
    "pgd"
]

TRAINING_MODE_TAGS = {
    "clean": "standard",
    "fgsm": "advtrain_fgsm",
    "pgd": "advtrain_pgd"
}

# Perturbation budget used while crafting TRAINING batches.
# Default: 8/255 ~= 0.031, the standard CIFAR-10 budget.
ADV_TRAIN_EPSILON = DATASET_CONFIG.get(
    "adv_train_epsilon",
    8.0 / 255.0
)

# Inner-maximization steps during PGD adversarial training.
ADV_TRAIN_PGD_STEPS = DATASET_CONFIG.get(
    "adv_train_pgd_steps",
    5
)

# Optional epoch override for adversarially-trained runs
# (None = same as standard training). Adversarial epochs are
# ~3x (FGSM) to ~6x (PGD-5) more expensive per epoch; lower
# this if CPU runtime becomes prohibitive.
ADV_TRAIN_EPOCHS_OVERRIDE = DATASET_CONFIG.get(
    "adv_train_epochs",
    None
)


# =========================================================
# RESULTS DIRECTORIES
#
# Per-training-mode subfolders keep every variant's outputs
# separate:
#
#   results/<dataset>/standard/metrics|plots        (clean)
#   results/<dataset>/advtrain_fgsm/metrics|plots
#   results/<dataset>/advtrain_pgd/metrics|plots
#
# plus dataset-level cross-mode comparison artifacts at
# results/<dataset>/{adversarial_training_comparison.csv,
# plots/adversarial_training_comparison.png}.
# =========================================================

RESULTS_DIR = os.path.join(
    ANALYSIS_DIR,
    "results",
    DATASET_NAME
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "standard",
    "metrics"
)

PLOTS_DIR = os.path.join(
    RESULTS_DIR,
    "standard",
    "plots"
)


def set_output_dirs(training_mode):
    """
    Point METRICS_DIR / PLOTS_DIR at the folder for the
    given training mode. Plot/save helpers read these
    module-level globals, so calling this before the save
    stage routes all artifacts of that run correctly.
    """

    global METRICS_DIR, PLOTS_DIR

    tag = TRAINING_MODE_TAGS[training_mode]

    METRICS_DIR = os.path.join(
        RESULTS_DIR,
        tag,
        "metrics"
    )

    PLOTS_DIR = os.path.join(
        RESULTS_DIR,
        tag,
        "plots"
    )

    os.makedirs(
        METRICS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )


os.makedirs(
    METRICS_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)

print(
    "ADVERSARIAL ATTACKS ANALYSIS"
)

print(
    "VIB with KL / Renyi / Tsallis Divergences"
)

print("=" * 60)

print(
    f"Device: {device}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Training modes: {', '.join(TRAINING_MODES)}"
)

print(
    f"Adv-training epsilon: "
    f"{ADV_TRAIN_EPSILON:.4f} (8/255 default)"
)

print(
    f"Adv-training PGD steps: {ADV_TRAIN_PGD_STEPS}"
)

if ADV_TRAIN_EPOCHS_OVERRIDE is not None:

    print(
        f"Adv-training epochs override: "
        f"{ADV_TRAIN_EPOCHS_OVERRIDE}"
    )

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Train subset: "
    f"{TRAIN_SUBSET_FRACTION:.0%}"
)

print(
    f"Test subset: "
    f"{TEST_SUBSET_FRACTION:.0%}"
)

print(
    f"Dataset: {DATASET_NAME}"
)

print(
    f"Architecture: {ARCHITECTURE}"
)

print(
    f"Input shape: {INPUT_SHAPE} "
    f"(flatten={FLATTEN_INPUTS})"
)

print(
    f"Num classes: {NUM_CLASSES}"
)

print(
    f"Beta: {BETA}"
)

print(
    f"Hidden dim: {HIDDEN_DIM}"
)

print(
    f"Latent dim: {LATENT_DIM}"
)

print(
    f"Seed: {SEED}"
)

print(
    f"Renyi alpha: {RENYI_ALPHA}"
)

print(
    f"Tsallis alpha: {TSALLIS_ALPHA}"
)

print(
    f"Attack epsilons: {EPSILONS}"
)

print(
    f"PGD steps: {PGD_STEPS}"
)

print("=" * 60)


# =========================================================
# DATASET
#
# Flattening is handled once, here, via collate_fn. Every
# downstream consumer (training loop, clean eval, attacks)
# receives batches already in the model's input format, so
# no other file needs dataset-specific shape logic.
# =========================================================

def make_collate(flatten):
    """
    Build a collate_fn that stacks a batch and optionally
    flattens images to (B, C*H*W) for MLP models.
    """

    def _collate(batch):

        images = torch.stack(
            [item[0] for item in batch]
        )

        labels = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.long
        )

        if flatten:

            images = images.view(
                images.size(0),
                -1
            )

        return images, labels

    return _collate


_collate_fn = make_collate(FLATTEN_INPUTS)


dataset_cls = DATASET_CONFIG["dataset_class"]


train_dataset = dataset_cls(
    root=os.path.join(
        ROOT_DIR,
        "data"
    ),
    train=True,
    download=True,
    transform=TRAIN_TRANSFORM
)


test_dataset = dataset_cls(
    root=os.path.join(
        ROOT_DIR,
        "data"
    ),
    train=False,
    download=True,
    transform=TEST_TRANSFORM
)


# ---------------------------------------------------------
# Reproducible subsampling (compute-load control)
#
# A seeded generator picks the subset indices, so the same
# fractions always yield the same splits across runs and
# across divergences — the comparison stays fair.
# ---------------------------------------------------------

def make_subset(dataset, fraction):

    if fraction >= 1.0:

        return dataset

    num_keep = int(
        len(dataset) * fraction
    )

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:num_keep].tolist()

    print(
        f"Subset: using {num_keep}/{len(dataset)} "
        f"samples ({fraction:.0%})"
    )

    return Subset(dataset, indices)


train_dataset = make_subset(
    train_dataset,
    TRAIN_SUBSET_FRACTION
)

test_dataset = make_subset(
    test_dataset,
    TEST_SUBSET_FRACTION
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=_collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=_collate_fn
)


# =========================================================
# TRAINING
# =========================================================

def train_model(
    model,
    divergence,
    training_mode="clean",
    epochs=None
):
    """
    Train one VIB model.

    training_mode selects the batch distribution:

        "clean" -> standard VIB training on clean images

        "fgsm"  -> FGSM-adversarial training: every batch is
                   replaced by a single-step adversarial
                   version (eps = ADV_TRAIN_EPSILON)

        "pgd"   -> PGD-adversarial training: every batch is
                   replaced by a multi-step PGD inner
                   maximization (Madry-style) at the same eps

    The divergence term is computed on the SAME forward pass
    used for classification (mu / logvar of the batch the CE
    loss sees), so the VIB objective stays consistent under
    all three modes.
    """

    if epochs is None:

        epochs = EPOCHS


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Cosine annealing: smooth decay to ~0 over training,
    # improves convergence on harder datasets.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    history = []

    train_start = time.time()

    epoch_times = []


    for epoch in range(epochs):

        epoch_start = time.time()

        model.train()

        total_loss = 0.0

        total_ce = 0.0

        total_information = 0.0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            # -------------------------------------------------
            # Adversarial training: replace the batch with an
            # adversarial version crafted on-the-fly. This
            # must happen BEFORE optimizer.zero_grad() — the
            # generator uses autograd.grad and never touches
            # parameter .grad buffers, so the only gradients
            # the optimizer sees are from the training loss
            # below.
            # -------------------------------------------------

            if training_mode != "clean":

                images = generate_adversarial_batch(
                    model,
                    images,
                    labels,
                    ADV_TRAIN_EPSILON,
                    device,
                    method=training_mode,
                    num_steps=ADV_TRAIN_PGD_STEPS
                )


            optimizer.zero_grad()


            logits, z, mu, logvar = model(
                images
            )


            # Classification loss
            classification_loss = F.cross_entropy(
                logits,
                labels
            )


            # Training divergence
            if divergence == "renyi":

                alpha = RENYI_ALPHA

            elif divergence == "tsallis":

                alpha = TSALLIS_ALPHA

            else:

                alpha = 0.5


            information_loss = divergence_loss(
                mu,
                logvar,
                divergence=divergence,
                alpha=alpha
            )


            # VIB objective
            loss = (
                classification_loss
                + BETA * information_loss
            )

            # Add weak KL regularizer to prevent vanishing gradients
            if divergence == "tsallis":
                kl_loss = divergence_loss(
                    mu,
                    logvar,
                    divergence="kl"
                )
                loss = loss + 1e-5 * kl_loss


            # Numerical sanity check
            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss encountered during "
                    f"{divergence} training."
                )


            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )


            optimizer.step()


            total_loss += loss.item()

            total_ce += (
                classification_loss.item()
            )

            total_information += (
                information_loss.item()
            )


        n_batches = len(
            train_loader
        )


        epoch_loss = (
            total_loss
            / n_batches
        )

        epoch_ce = (
            total_ce
            / n_batches
        )

        epoch_information = (
            total_information
            / n_batches
        )


        epoch_time = time.time() - epoch_start

        epoch_times.append(epoch_time)

        elapsed = time.time() - train_start

        avg_epoch = (
            sum(epoch_times) / len(epoch_times)
        )

        eta_seconds = (
            avg_epoch * (epochs - epoch - 1)
        )

        eta_minutes = eta_seconds / 60.0


        history.append({

            "epoch":
                epoch + 1,

            "loss":
                epoch_loss,

            "cross_entropy":
                epoch_ce,

            "information_loss":
                epoch_information
        })


        print(
            f"[{divergence.upper():7s}] "
            f"[train: {training_mode.upper()}] "
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"CE: {epoch_ce:.4f} | "
            f"Info: {epoch_information:.4f} | "
            f"Time: {epoch_time:.1f}s | "
            f"ETA: {eta_minutes:.1f} min",
            flush=True
        )


        scheduler.step()


    total_train_minutes = (
        time.time() - train_start
    ) / 60.0

    print(
        f"[{divergence.upper():7s}] "
        f"Training finished in "
        f"{total_train_minutes:.1f} min",
        flush=True
    )


    return history


# =========================================================
# CLEAN EVALUATION
# =========================================================

def evaluate_clean(
    model,
    divergence
):
    """
    Evaluate clean (non-adversarial) accuracy.

    Uses deterministic forward pass (encoder mean,
    no sampling).
    """

    model.eval()

    correct = 0
    total = 0

    all_mu = []
    all_logvar = []
    all_logits = []
    all_labels = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)


            mu, logvar = model.encode(images)

            logits = model.classifier(mu)


            predictions = torch.argmax(
                logits,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())


    accuracy = correct / total

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    return {
        "accuracy": accuracy,
        "mu": all_mu,
        "logvar": all_logvar,
        "logits": all_logits,
        "labels": all_labels
    }


# =========================================================
# ADVERSARIAL EVALUATION
# =========================================================

def run_adversarial_evaluation(
    model,
    divergence_name
):
    """
    Run FGSM and PGD attacks at all epsilon values.

    Returns a dict of results.
    """

    results = {}


    for epsilon in EPSILONS:

        print(
            f"\n  [{divergence_name.upper()}] "
            f"FGSM epsilon={epsilon:.2f} ...",
            flush=True
        )

        fgsm_accuracy = evaluate_under_attack(
            model,
            fgsm_attack,
            test_loader,
            epsilon,
            device
        )

        print(
            f"    FGSM accuracy: "
            f"{fgsm_accuracy:.4f}"
        )


        print(
            f"  [{divergence_name.upper()}] "
            f"PGD epsilon={epsilon:.2f} "
            f"(steps={PGD_STEPS}) ...",
            flush=True
        )

        def pgd_attack_fn(
            model, images, labels, eps, dev
        ):
            return pgd_attack(
                model,
                images,
                labels,
                eps,
                dev,
                num_steps=PGD_STEPS,
                step_size=PGD_STEP_SIZE
            )

        pgd_accuracy = evaluate_under_attack(
            model,
            pgd_attack_fn,
            test_loader,
            epsilon,
            device
        )

        print(
            f"    PGD accuracy: "
            f"{pgd_accuracy:.4f}"
        )


        results[epsilon] = {
            "fgsm_accuracy": fgsm_accuracy,
            "pgd_accuracy": pgd_accuracy
        }


    return results


# =========================================================
# PLOTTING
# =========================================================

def plot_clean_accuracy(
    all_results
):
    """Bar chart of clean accuracy per divergence."""

    methods = list(all_results.keys())

    values = [
        all_results[m]["clean_accuracy"]
        for m in methods
    ]


    plt.figure(figsize=(7, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    plt.bar(
        [m.upper() for m in methods],
        values,
        color=colors[:len(methods)]
    )

    plt.ylabel("Accuracy")
    plt.title("Clean Accuracy by Divergence")
    plt.ylim(0, 1.0)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "clean_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


def plot_robustness_curves(
    all_results,
    attack_name
):
    """
    Line plot: accuracy vs epsilon for a given attack.
    One line per divergence.
    """

    plt.figure(figsize=(8, 5))

    colors = {
        "kl": "#2196F3",
        "renyi": "#FF9800",
        "tsallis": "#4CAF50"
    }

    markers = {
        "kl": "o",
        "renyi": "s",
        "tsallis": "^"
    }

    key = f"{attack_name}_accuracy"


    for method in all_results:

        epsilons = sorted(
            all_results[method]["adversarial"].keys()
        )

        accuracies = [
            all_results[method]["adversarial"][eps][key]
            for eps in epsilons
        ]

        plt.plot(
            epsilons,
            accuracies,
            marker=markers.get(method, "o"),
            color=colors.get(method, "#000000"),
            label=method.upper(),
            linewidth=2,
            markersize=8
        )


    plt.xlabel("Epsilon (Perturbation Strength)")
    plt.ylabel("Accuracy")
    plt.title(
        f"{attack_name.upper()} Adversarial Robustness"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{attack_name}_robustness.png"
        ),
        dpi=300
    )

    plt.close()


def plot_robustness_vs_ixz(
    all_results
):
    """
    Scatter: adversarial accuracy (at mid-epsilon)
    vs I(X;Z).
    """

    mid_epsilon = EPSILONS[len(EPSILONS) // 2]

    plt.figure(figsize=(8, 5))

    colors = {
        "kl": "#2196F3",
        "renyi": "#FF9800",
        "tsallis": "#4CAF50"
    }


    for method in all_results:

        i_xz = all_results[method]["I_XZ"]

        fgsm_acc = (
            all_results[method]
            ["adversarial"]
            [mid_epsilon]
            ["fgsm_accuracy"]
        )

        pgd_acc = (
            all_results[method]
            ["adversarial"]
            [mid_epsilon]
            ["pgd_accuracy"]
        )


        plt.scatter(
            i_xz,
            fgsm_acc,
            marker="o",
            color=colors.get(method, "#000"),
            s=120,
            label=f"{method.upper()} FGSM"
        )

        plt.scatter(
            i_xz,
            pgd_acc,
            marker="^",
            color=colors.get(method, "#000"),
            s=120,
            label=f"{method.upper()} PGD"
        )

        plt.annotate(
            method.upper(),
            (i_xz, fgsm_acc),
            textcoords="offset points",
            xytext=(8, 8)
        )


    plt.xlabel("I(X;Z)")
    plt.ylabel(
        f"Adversarial Accuracy (epsilon={mid_epsilon})"
    )
    plt.title(
        "Adversarial Robustness vs Information Compression"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "robustness_vs_ixz.png"
        ),
        dpi=300
    )

    plt.close()


def plot_ib_tradeoff(
    all_results
):
    """
    Scatter: I(X;Z) vs I(Z;Y) annotated with
    robustness info.
    """

    mid_epsilon = EPSILONS[len(EPSILONS) // 2]

    plt.figure(figsize=(8, 5))

    colors = {
        "kl": "#2196F3",
        "renyi": "#FF9800",
        "tsallis": "#4CAF50"
    }


    for method in all_results:

        i_xz = all_results[method]["I_XZ"]
        i_zy = all_results[method]["I_ZY"]

        pgd_acc = (
            all_results[method]
            ["adversarial"]
            [mid_epsilon]
            ["pgd_accuracy"]
        )


        plt.scatter(
            i_xz,
            i_zy,
            color=colors.get(method, "#000"),
            s=200,
            label=method.upper(),
            zorder=5
        )

        plt.annotate(
            f"{method.upper()}\n"
            f"PGD@{mid_epsilon}: "
            f"{pgd_acc:.2%}",
            (i_xz, i_zy),
            textcoords="offset points",
            xytext=(12, -5),
            fontsize=9
        )


    plt.xlabel("I(X;Z)")
    plt.ylabel("I(Z;Y)")
    plt.title(
        "Information Bottleneck Trade-off "
        "with Adversarial Robustness"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "ib_tradeoff.png"
        ),
        dpi=300
    )

    plt.close()


def plot_summary_table(
    all_results
):
    """
    Render a summary table as a figure.
    """

    mid_epsilon = EPSILONS[len(EPSILONS) // 2]

    rows = []

    for method in all_results:

        row = {
            "Divergence": method.upper(),

            "Clean Acc": (
                f"{all_results[method]['clean_accuracy']:.4f}"
            ),

            "I(X;Z)": (
                f"{all_results[method]['I_XZ']:.3f}"
            ),

            "I(Z;Y)": (
                f"{all_results[method]['I_ZY']:.3f}"
            ),

            f"FGSM@{mid_epsilon}": (
                f"{all_results[method]['adversarial'][mid_epsilon]['fgsm_accuracy']:.4f}"
            ),

            f"PGD@{mid_epsilon}": (
                f"{all_results[method]['adversarial'][mid_epsilon]['pgd_accuracy']:.4f}"
            )
        }

        rows.append(row)


    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(12, 3)
    )

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Header styling
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(
            color="white",
            fontweight="bold"
        )

    plt.title(
        "Summary: Adversarial Robustness Analysis",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "summary_table.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def run_mode(training_mode):
    """
    Run the full pipeline (train all divergences, evaluate,
    save metrics and plots) for ONE training mode. All
    artifacts are routed to that mode's subfolder under
    results/<dataset>/.
    """

    run_start = time.time()

    set_seed(SEED)

    set_output_dirs(training_mode)

    tag = TRAINING_MODE_TAGS[training_mode]


    print("\n")
    print("#" * 60)
    print(
        f"TRAINING MODE: {training_mode.upper()} "
        f"(results -> results/{DATASET_NAME}/{tag}/)"
    )
    if training_mode != "clean":

        print(
            f"  Adversarial training eps: "
            f"{ADV_TRAIN_EPSILON:.4f}"
        )
        if training_mode == "pgd":

            print(
                f"  Adversarial training PGD steps: "
                f"{ADV_TRAIN_PGD_STEPS}"
            )

    print("#" * 60)


    # -----------------------------------------------------
    # Epoch budget for this mode (optional override for the
    # costlier adversarial variants).
    # -----------------------------------------------------

    if training_mode == "clean":

        epochs_for_mode = EPOCHS

    elif ADV_TRAIN_EPOCHS_OVERRIDE is not None:

        epochs_for_mode = ADV_TRAIN_EPOCHS_OVERRIDE

    else:

        epochs_for_mode = EPOCHS


    all_results = {}

    histories = {}


    for divergence in DIVERGENCES:

        divergence_start = time.time()

        print("\n")

        print("=" * 60)

        print(
            f"TRAINING {divergence.upper()} VIB "
            f"[{training_mode.upper()}]"
        )

        print("=" * 60)


        # -------------------------------------------------
        # Fresh model
        # -------------------------------------------------

        model = build_vib(
            arch=ARCHITECTURE,
            input_shape=INPUT_SHAPE,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            num_classes=NUM_CLASSES
        ).to(device)


        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        history = train_model(
            model,
            divergence,
            training_mode=training_mode,
            epochs=epochs_for_mode
        )

        histories[divergence] = history


        # -------------------------------------------------
        # Clean evaluation
        # -------------------------------------------------

        clean_eval_start = time.time()

        print(
            f"\n  Evaluating clean accuracy ...",
            flush=True
        )

        clean_eval = evaluate_clean(
            model,
            divergence
        )

        clean_eval_minutes = (
            time.time() - clean_eval_start
        ) / 60.0

        print(
            f"  Clean accuracy: "
            f"{clean_eval['accuracy']:.4f} "
            f"({clean_eval_minutes:.1f} min)",
            flush=True
        )


        # -------------------------------------------------
        # Information metrics (KL-based for all)
        # -------------------------------------------------

        info_metrics = (
            calculate_all_information_metrics(

                mu=clean_eval["mu"],

                logvar=clean_eval["logvar"],

                labels=clean_eval["labels"],

                logits=clean_eval["logits"],

                num_classes=NUM_CLASSES,

                beta=BETA
            )
        )

        print(
            f"  I(X;Z): "
            f"{info_metrics['I_XZ']:.4f}"
        )

        print(
            f"  I(Z;Y): "
            f"{info_metrics['I_ZY']:.4f}"
        )


        # -------------------------------------------------
        # Adversarial evaluation
        # -------------------------------------------------

        adversarial_start = time.time()

        print(
            f"\n  Running adversarial attacks ...",
            flush=True
        )

        adversarial_results = (
            run_adversarial_evaluation(
                model,
                divergence
            )
        )

        adversarial_minutes = (
            time.time() - adversarial_start
        ) / 60.0

        divergence_minutes = (
            time.time() - divergence_start
        ) / 60.0

        elapsed_minutes = (
            time.time() - run_start
        ) / 60.0


        print(
            f"\n  [{divergence.upper()}] "
            f"Adversarial eval: "
            f"{adversarial_minutes:.1f} min | "
            f"Divergence total: "
            f"{divergence_minutes:.1f} min | "
            f"Overall elapsed: "
            f"{elapsed_minutes:.1f} min",
            flush=True
        )


        # -------------------------------------------------
        # Combine results
        # -------------------------------------------------

        all_results[divergence] = {

            "clean_accuracy":
                clean_eval["accuracy"],

            "I_XZ":
                info_metrics["I_XZ"],

            "H_Y":
                info_metrics["H_Y"],

            "H_Y_given_Z":
                info_metrics["H_Y_given_Z"],

            "I_ZY":
                info_metrics["I_ZY"],

            "IB_objective":
                info_metrics["IB_objective"],

            "adversarial":
                adversarial_results
        }


    # =====================================================
    # SAVE ADVERSARIAL RESULTS CSV
    # =====================================================

    print("\n")
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)


    adv_rows = []

    for method in all_results:

        for epsilon in EPSILONS:

            adv_rows.append({
                "divergence": method,
                "epsilon": epsilon,
                "clean_accuracy": (
                    all_results[method][
                        "clean_accuracy"
                    ]
                ),
                "fgsm_accuracy": (
                    all_results[method]
                    ["adversarial"]
                    [epsilon]
                    ["fgsm_accuracy"]
                ),
                "pgd_accuracy": (
                    all_results[method]
                    ["adversarial"]
                    [epsilon]
                    ["pgd_accuracy"]
                ),
                "I_XZ": (
                    all_results[method]["I_XZ"]
                ),
                "I_ZY": (
                    all_results[method]["I_ZY"]
                )
            })


    adv_df = pd.DataFrame(adv_rows)

    adv_csv_path = os.path.join(
        METRICS_DIR,
        "adversarial_results.csv"
    )

    adv_df.to_csv(
        adv_csv_path,
        index=False
    )

    print(
        f"\nAdversarial results saved: "
        f"{adv_csv_path}"
    )


    # =====================================================
    # SAVE INFORMATION METRICS CSV
    # =====================================================

    info_rows = []

    for method in all_results:

        info_rows.append({
            "divergence": method,
            "clean_accuracy": (
                all_results[method]["clean_accuracy"]
            ),
            "I_XZ": (
                all_results[method]["I_XZ"]
            ),
            "H_Y": (
                all_results[method]["H_Y"]
            ),
            "H_Y_given_Z": (
                all_results[method]["H_Y_given_Z"]
            ),
            "I_ZY": (
                all_results[method]["I_ZY"]
            ),
            "IB_objective": (
                all_results[method]["IB_objective"]
            )
        })


    info_df = pd.DataFrame(info_rows)

    info_csv_path = os.path.join(
        METRICS_DIR,
        "information_metrics.csv"
    )

    info_df.to_csv(
        info_csv_path,
        index=False
    )

    print(
        f"Information metrics saved: "
        f"{info_csv_path}"
    )


    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print("\n")
    print("=" * 60)
    print("ADVERSARIAL ROBUSTNESS SUMMARY")
    print("=" * 60)

    print(
        adv_df.to_string(
            index=False
        )
    )


    # =====================================================
    # PLOTS
    # =====================================================

    print("\n\nGenerating plots ...")


    plot_clean_accuracy(
        all_results
    )

    plot_robustness_curves(
        all_results,
        "fgsm"
    )

    plot_robustness_curves(
        all_results,
        "pgd"
    )

    plot_robustness_vs_ixz(
        all_results
    )

    plot_ib_tradeoff(
        all_results
    )

    plot_summary_table(
        all_results
    )


    print(
        f"\nPlots saved to: {PLOTS_DIR}"
    )

    print(
        f"Metrics saved to: {METRICS_DIR}"
    )

    mode_minutes = (
        time.time() - run_start
    ) / 60.0

    print(
        f"\n[{training_mode.upper()}] "
        f"Mode runtime: "
        f"{mode_minutes:.1f} min "
        f"({mode_minutes / 60.0:.2f} h)"
    )

    print(
        f"\nTraining mode '{training_mode}' completed."
    )


    return all_results


# =========================================================
# CROSS-MODE COMPARISON (clean vs FGSM-adv vs PGD-adv)
# =========================================================

def save_adversarial_training_comparison(
    mode_summaries
):
    """
    Aggregate the per-mode results into one table answering:

        "Does adversarial training help, and which inner
         maximization (FGSM vs PGD) helps more?"

    For every divergence x training mode it reports clean
    accuracy plus FGSM/PGD attack accuracy at the reference
    epsilon (the evaluation epsilon closest to the training
    budget ADV_TRAIN_EPSILON).
    """

    reference_epsilon = min(
        EPSILONS,
        key=lambda e: abs(e - ADV_TRAIN_EPSILON)
    )

    rows = []

    for training_mode in TRAINING_MODES:

        results = mode_summaries.get(training_mode)

        if results is None:

            continue

        for divergence in DIVERGENCES:

            if divergence not in results:

                continue

            entry = results[divergence]

            rows.append({
                "training_mode": (
                    TRAINING_MODE_TAGS[training_mode]
                ),
                "divergence": divergence,
                "train_epsilon": (
                    ADV_TRAIN_EPSILON
                    if training_mode != "clean"
                    else 0.0
                ),
                "clean_accuracy": (
                    entry["clean_accuracy"]
                ),
                "fgsm_accuracy": (
                    entry["adversarial"]
                    [reference_epsilon]
                    ["fgsm_accuracy"]
                ),
                "pgd_accuracy": (
                    entry["adversarial"]
                    [reference_epsilon]
                    ["pgd_accuracy"]
                )
            })


    comparison_df = pd.DataFrame(rows)

    comparison_csv_path = os.path.join(
        RESULTS_DIR,
        "adversarial_training_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_csv_path,
        index=False
    )

    print("\n")
    print("=" * 60)
    print(
        "ADVERSARIAL TRAINING COMPARISON "
        f"(eval epsilon = {reference_epsilon:.3f})"
    )
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(
        f"\nComparison CSV saved: "
        f"{comparison_csv_path}"
    )


    # -----------------------------------------------------
    # Grouped bar chart: robustness by training mode
    # -----------------------------------------------------

    os.makedirs(
        os.path.join(RESULTS_DIR, "plots"),
        exist_ok=True
    )

    tags = [
        TRAINING_MODE_TAGS[m]
        for m in TRAINING_MODES
        if m in mode_summaries
    ]

    bar_colors = {
        "standard": "#90A4AE",
        "advtrain_fgsm": "#FF9800",
        "advtrain_pgd": "#4CAF50"
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    for ax, attack in zip(
        axes,
        ["fgsm_accuracy", "pgd_accuracy"]
    ):

        width = 0.25

        for i, tag in enumerate(tags):

            values = [

                float(
                    comparison_df[

                        (comparison_df["training_mode"] == tag)

                        & (comparison_df["divergence"] == div)

                    ][attack].iloc[0]
                )

                for div in DIVERGENCES
                if not comparison_df[

                    (comparison_df["training_mode"] == tag)

                    & (comparison_df["divergence"] == div)

                ].empty
            ]

            labels = [
                div for div in DIVERGENCES
                if not comparison_df[

                    (comparison_df["training_mode"] == tag)

                    & (comparison_df["divergence"] == div)

                ].empty
            ]

            positions = [
                j + i * width - width
                for j in range(len(values))
            ]

            ax.bar(
                positions,
                values,
                width=width,
                label=tag,
                color=bar_colors.get(tag)
            )

        ax.set_xticks(range(len(DIVERGENCES)))

        ax.set_xticklabels(
            [d.upper() for d in DIVERGENCES]
        )

        ax.set_ylim(0, 1.0)

        ax.set_ylabel("Accuracy")

        ax.set_title(
            f"Robustness under {attack.split('_')[0].upper()} "
            f"(eps={reference_epsilon:.3f})"
        )

        ax.legend()

    plt.tight_layout()

    comparison_plot_path = os.path.join(
        RESULTS_DIR,
        "plots",
        "adversarial_training_comparison.png"
    )

    plt.savefig(
        comparison_plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Comparison plot saved: "
        f"{comparison_plot_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    overall_start = time.time()

    print("\n")
    print("=" * 60)
    print(
        f"ADVERSARIAL TRAINING PIPELINE | modes: "
        f"{', '.join(TRAINING_MODES)}"
    )
    print("=" * 60)


    mode_summaries = {}

    for training_mode in TRAINING_MODES:

        mode_summaries[training_mode] = run_mode(
            training_mode
        )


    save_adversarial_training_comparison(
        mode_summaries
    )

    total_minutes = (
        time.time() - overall_start
    ) / 60.0

    print(
        f"\nTotal pipeline runtime: "
        f"{total_minutes:.1f} min "
        f"({total_minutes / 60.0:.2f} h)"
    )

    print(
        "\nAdversarial attacks analysis completed."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()