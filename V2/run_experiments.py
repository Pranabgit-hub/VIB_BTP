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


# ---------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# V2 directory itself
V2_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, V2_DIR)


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
BETA = 1e-3

LATENT_DIM = 32

RENYI_ALPHA = 2.0
TSALLIS_ALPHA = 2.0

# Same single seed as V1
SEED = 42

DIVERGENCES = [
    "kl",
    "renyi",
    "tsallis"
]


# ---------------------------------------------------------
# V2 Results Directory
# ---------------------------------------------------------

# IMPORTANT:
# Results are stored inside v2/results,
# NOT the V1 results directory.

RESULTS_DIR = os.path.join(
    V2_DIR,
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


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("VIB VERSION 2 EXPERIMENT")
print("=" * 60)

print(f"Device: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Beta: {BETA}")
print(f"Seed: {SEED}")
print(f"Renyi alpha: {RENYI_ALPHA}")
print(f"Tsallis alpha: {TSALLIS_ALPHA}")

print("=" * 60)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_model(model, divergence):

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

            logits, z, mu, logvar = model(images)

            classification_loss = F.cross_entropy(
                logits,
                labels
            )

            # -------------------------------------------------
            # V2:
            # Selected divergence is used during training.
            #
            # KL      -> exact Gaussian KL
            # Renyi   -> exact Gaussian Renyi
            # Tsallis -> exact Gaussian Tsallis
            # -------------------------------------------------

            information_loss = divergence_loss(
                mu,
                logvar,
                divergence=divergence,
                alpha=(
                    RENYI_ALPHA
                    if divergence == "renyi"
                    else TSALLIS_ALPHA
                )
            )

            loss = (
                classification_loss
                + BETA * information_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_ce += classification_loss.item()
            total_information += information_loss.item()

        n_batches = len(train_loader)

        epoch_loss = (
            total_loss / n_batches
        )

        epoch_ce = (
            total_ce / n_batches
        )

        epoch_information = (
            total_information / n_batches
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "cross_entropy": epoch_ce,
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


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(model, divergence):

    model.eval()

    correct = 0
    total = 0

    total_ce = 0.0
    total_information = 0.0

    latent_samples = []

    # Store values needed for I(X;Z), H(Y|Z), etc.
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

            logits, z, mu, logvar = model(images)

            ce = F.cross_entropy(
                logits,
                labels
            )

            # -------------------------------------------------
            # Training divergence evaluated on test data
            # -------------------------------------------------

            info = divergence_loss(
                mu,
                logvar,
                divergence=divergence,
                alpha=(
                    RENYI_ALPHA
                    if divergence == "renyi"
                    else TSALLIS_ALPHA
                )
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            total_ce += ce.item()

            total_information += (
                info.item()
            )

            latent_samples.append(
                z.cpu()
            )

            all_mu.append(
                mu.cpu()
            )

            all_logvar.append(
                logvar.cpu()
            )

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    accuracy = (
        correct / total
    )

    latent_samples = torch.cat(
        latent_samples,
        dim=0
    )

    all_mu = torch.cat(
        all_mu,
        dim=0
    )

    all_logvar = torch.cat(
        all_logvar,
        dim=0
    )

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    return {
        "accuracy":
            accuracy,

        "cross_entropy":
            total_ce / len(test_loader),

        "information_loss":
            total_information
            / len(test_loader),

        "latent":
            latent_samples,

        "mu":
            all_mu,

        "logvar":
            all_logvar,

        "logits":
            all_logits,

        "labels":
            all_labels
    }


# ---------------------------------------------------------
# Entropy estimation
# ---------------------------------------------------------

# KEEPING THE V1 HISTOGRAM METHOD
# No new entropy implementation.

def estimate_entropy(latent):

    """
    Histogram-based entropy estimation.

    The continuous latent representation Z is discretized
    into histogram bins before calculating:

        Shannon entropy
        Renyi entropy
        Tsallis entropy
    """

    values = latent.numpy().flatten()

    hist, _ = np.histogram(
        values,
        bins=100,
        density=False
    )

    probabilities = (
        hist / hist.sum()
    )

    probabilities = torch.tensor(
        probabilities,
        dtype=torch.float32
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    shannon = shannon_entropy(
        probabilities
    ).item()

    renyi = renyi_entropy(
        probabilities,
        alpha=RENYI_ALPHA
    ).item()

    tsallis = tsallis_entropy(
        probabilities,
        alpha=TSALLIS_ALPHA
    ).item()

    return {
        "shannon_entropy":
            shannon,

        "renyi_entropy":
            renyi,

        "tsallis_entropy":
            tsallis
    }


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_bar(
    results,
    metric,
    title,
    ylabel,
    filename
):

    methods = list(
        results.keys()
    )

    values = [
        results[m][metric]
        for m in methods
    ]

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        methods,
        values
    )

    plt.xlabel(
        "Divergence"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


def plot_accuracy_vs_information(
    results
):

    plt.figure(
        figsize=(7, 5)
    )

    for method in results:

        x = results[method][
            "I_XZ"
        ]

        y = results[method][
            "accuracy"
        ]

        plt.scatter(
            x,
            y,
            s=100,
            label=method.upper()
        )

        plt.annotate(
            method.upper(),
            (x, y)
        )

    plt.xlabel(
        "I(X;Z)"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Accuracy vs I(X;Z)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "accuracy_vs_I_XZ.png"
        ),
        dpi=300
    )

    plt.close()


def plot_ib_tradeoff(results):

    plt.figure(
        figsize=(7, 5)
    )

    for method in results:

        x = results[method][
            "I_XZ"
        ]

        y = results[method][
            "I_ZY"
        ]

        plt.scatter(
            x,
            y,
            s=100,
            label=method.upper()
        )

        plt.annotate(
            method.upper(),
            (x, y)
        )

    plt.xlabel(
        "I(X;Z)"
    )

    plt.ylabel(
        "I(Z;Y)"
    )

    plt.title(
        "Information Bottleneck Trade-off"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "information_bottleneck_tradeoff.png"
        ),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    set_seed(SEED)

    results = {}

    histories = {}

    for divergence in DIVERGENCES:

        print("\n")
        print("=" * 60)

        print(
            f"TRAINING {divergence.upper()} VIB"
        )

        print("=" * 60)

        # -------------------------------------------------
        # Fresh model for each divergence
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

        histories[
            divergence
        ] = history

        # -------------------------------------------------
        # Evaluate
        # -------------------------------------------------

        evaluation = evaluate_model(
            model,
            divergence
        )

        # -------------------------------------------------
        # Histogram entropy
        # -------------------------------------------------

        entropy_results = estimate_entropy(
            evaluation["latent"]
        )

        # -------------------------------------------------
        # Information-theoretic metrics
        #
        # I(X;Z) is evaluated using KL for ALL models.
        # -------------------------------------------------

        information_results = (
            calculate_all_information_metrics(
                mu=evaluation["mu"],
                logvar=evaluation["logvar"],
                labels=evaluation["labels"],
                logits=evaluation["logits"],
                num_classes=10,
                beta=BETA
            )
        )

        # -------------------------------------------------
        # Combine all metrics
        # -------------------------------------------------

        results[divergence] = {

            # Standard performance
            "accuracy":
                evaluation[
                    "accuracy"
                ],

            "cross_entropy":
                evaluation[
                    "cross_entropy"
                ],

            # Divergence actually used
            # during training
            "information_loss":
                evaluation[
                    "information_loss"
                ],

            # Histogram entropy
            "shannon_entropy":
                entropy_results[
                    "shannon_entropy"
                ],

            "renyi_entropy":
                entropy_results[
                    "renyi_entropy"
                ],

            "tsallis_entropy":
                entropy_results[
                    "tsallis_entropy"
                ],

            # Information Bottleneck metrics
            "I_XZ":
                information_results[
                    "I_XZ"
                ],

            "H_Y":
                information_results[
                    "H_Y"
                ],

            "H_Y_given_Z":
                information_results[
                    "H_Y_given_Z"
                ],

            "I_ZY":
                information_results[
                    "I_ZY"
                ],

            "IB_objective":
                information_results[
                    "IB_objective"
                ]
        }

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print("\nResults:")

        print(
            f"Accuracy: "
            f"{evaluation['accuracy']:.4f}"
        )

        print(
            f"Cross Entropy: "
            f"{evaluation['cross_entropy']:.4f}"
        )

        print(
            f"Training Divergence: "
            f"{evaluation['information_loss']:.4f}"
        )

        print(
            f"Shannon Entropy: "
            f"{entropy_results['shannon_entropy']:.4f}"
        )

        print(
            f"Renyi Entropy: "
            f"{entropy_results['renyi_entropy']:.4f}"
        )

        print(
            f"Tsallis Entropy: "
            f"{entropy_results['tsallis_entropy']:.4f}"
        )

        print(
            f"I(X;Z): "
            f"{information_results['I_XZ']:.4f}"
        )

        print(
            f"H(Y): "
            f"{information_results['H_Y']:.4f}"
        )

        print(
            f"H(Y|Z): "
            f"{information_results['H_Y_given_Z']:.4f}"
        )

        print(
            f"I(Z;Y): "
            f"{information_results['I_ZY']:.4f}"
        )

        print(
            f"IB Objective: "
            f"{information_results['IB_objective']:.4f}"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    dataframe = pd.DataFrame(
        results
    ).T

    dataframe.index.name = "divergence"

    csv_path = os.path.join(
        METRICS_DIR,
        "results_v2.csv"
    )

    dataframe.to_csv(
        csv_path
    )

    print("\n")
    print("=" * 60)
    print("V2 FINAL RESULTS")
    print("=" * 60)

    print(dataframe)

    print("\nSaved:")
    print(csv_path)

    # -----------------------------------------------------
    # Generate graphs
    # -----------------------------------------------------

    plot_bar(
        results,
        "accuracy",
        "V2 VIB Classification Accuracy",
        "Accuracy",
        "accuracy_v2.png"
    )

    plot_bar(
        results,
        "cross_entropy",
        "V2 VIB Cross Entropy",
        "Cross Entropy",
        "cross_entropy_v2.png"
    )

    plot_bar(
        results,
        "information_loss",
        "Training Divergence",
        "Divergence",
        "training_divergence_v2.png"
    )

    plot_bar(
        results,
        "shannon_entropy",
        "Shannon Entropy of Latent Representation",
        "Shannon Entropy",
        "shannon_entropy_v2.png"
    )

    plot_bar(
        results,
        "renyi_entropy",
        "Renyi Entropy of Latent Representation",
        "Renyi Entropy",
        "renyi_entropy_v2.png"
    )

    plot_bar(
        results,
        "tsallis_entropy",
        "Tsallis Entropy of Latent Representation",
        "Tsallis Entropy",
        "tsallis_entropy_v2.png"
    )

    plot_bar(
        results,
        "I_XZ",
        "I(X;Z) - KL Evaluation",
        "I(X;Z)",
        "I_XZ_v2.png"
    )

    plot_bar(
        results,
        "I_ZY",
        "I(Z;Y)",
        "I(Z;Y)",
        "I_ZY_v2.png"
    )

    plot_bar(
        results,
        "IB_objective",
        "Information Bottleneck Objective",
        "I(X;Z) - Beta I(Z;Y)",
        "IB_objective_v2.png"
    )

    plot_accuracy_vs_information(
        results
    )

    plot_ib_tradeoff(
        results
    )

    print("\nGraphs saved to:")
    print(PLOTS_DIR)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()