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
    evaluate_under_attack
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

BETA = 1e-3

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


# =========================================================
# RESULTS DIRECTORIES
# =========================================================

RESULTS_DIR = os.path.join(
    ANALYSIS_DIR,
    "results",
    DATASET_NAME
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

PLOTS_DIR = os.path.join(
    RESULTS_DIR,
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
    divergence
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Cosine annealing: smooth decay to ~0 over training,
    # improves convergence on harder datasets.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    history = []

    train_start = time.time()

    epoch_times = []


    for epoch in range(EPOCHS):

        epoch_start = time.time()

        model.train()

        total_loss = 0.0

        total_ce = 0.0

        total_information = 0.0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


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
            avg_epoch * (EPOCHS - epoch - 1)
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
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
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

def main():

    run_start = time.time()

    set_seed(SEED)


    all_results = {}

    histories = {}


    for divergence in DIVERGENCES:

        divergence_start = time.time()

        print("\n")

        print("=" * 60)

        print(
            f"TRAINING {divergence.upper()} VIB"
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
            divergence
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

    total_minutes = (
        time.time() - run_start
    ) / 60.0

    print(
        f"\nTotal runtime: "
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