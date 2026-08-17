import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
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

from model import VIB

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
# =========================================================

BATCH_SIZE = 128

EPOCHS = 20

LEARNING_RATE = 1e-3

BETA = 1e-3

LATENT_DIM = 32


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
# ---------------------------------------------------------

EPSILONS = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.3
]

PGD_STEPS = 20

PGD_STEP_SIZE = None  # defaults to epsilon / 4


# =========================================================
# RESULTS DIRECTORIES
# =========================================================

RESULTS_DIR = os.path.join(
    ANALYSIS_DIR,
    "results"
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
    f"Beta: {BETA}"
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
# =========================================================

transform = transforms.ToTensor()


train_dataset = datasets.MNIST(
    root=os.path.join(
        ROOT_DIR,
        "data"
    ),
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root=os.path.join(
        ROOT_DIR,
        "data"
    ),
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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

    history = []


    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        total_ce = 0.0

        total_information = 0.0


        for images, labels in train_loader:

            images = images.view(
                images.size(0),
                -1
            ).to(device)

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
            f"Info: {epoch_information:.4f}"
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

            images = images.view(
                images.size(0),
                -1
            ).to(device)

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
            f"FGSM epsilon={epsilon:.2f} ..."
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
            f"(steps={PGD_STEPS}) ..."
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
    plt.ylim(0.9, 1.0)

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

    set_seed(SEED)


    all_results = {}

    histories = {}


    for divergence in DIVERGENCES:

        print("\n")

        print("=" * 60)

        print(
            f"TRAINING {divergence.upper()} VIB"
        )

        print("=" * 60)


        # -------------------------------------------------
        # Fresh model
        # -------------------------------------------------

        model = VIB(
            latent_dim=LATENT_DIM
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

        print(
            f"\n  Evaluating clean accuracy ..."
        )

        clean_eval = evaluate_clean(
            model,
            divergence
        )

        print(
            f"  Clean accuracy: "
            f"{clean_eval['accuracy']:.4f}"
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

                num_classes=10,

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

        print(
            f"\n  Running adversarial attacks ..."
        )

        adversarial_results = (
            run_adversarial_evaluation(
                model,
                divergence
            )
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

    print(
        "\nAdversarial attacks analysis completed."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
