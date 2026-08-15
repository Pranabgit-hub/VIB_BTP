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
# Make V1/src importable
# ---------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import through the src package so that losses.py can keep
# its relative import: from .divergences import ...
from src.model import VIB
from src.losses import divergence_loss
from src.entropy import (
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy
)

# ---------------------------------------------------------
# Import V1 information-analysis functionality
#
# IMPORTANT:
# We import the metric function from information_analysis.py.
# information_analysis.py has its main() protected by:
#
#     if __name__ == "__main__":
#         main()
#
# Therefore importing it does NOT train another model.
# We only reuse its information-metric implementation.
# ---------------------------------------------------------

from information_analysis import (
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

SEED = 42

DIVERGENCES = [
    "kl",
    "renyi",
    "tsallis"
]

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
INFORMATION_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "information_analysis"
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(INFORMATION_RESULTS_DIR, exist_ok=True)


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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("VIB EXPERIMENT")
print("=" * 60)
print(f"Device: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Beta: {BETA}")
print("=" * 60)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=os.path.join(ROOT_DIR, "data"),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=os.path.join(ROOT_DIR, "data"),
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
                images.size(0), -1
            ).to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            logits, z, mu, logvar = model(images)

            classification_loss = F.cross_entropy(
                logits,
                labels
            )

            information_loss = divergence_loss(
                mu,
                logvar,
                divergence
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

        epoch_loss = total_loss / n_batches
        epoch_ce = total_ce / n_batches
        epoch_information = (
            total_information / n_batches
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "cross_entropy": epoch_ce,
            "information_loss": epoch_information
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

    # Additional tensors needed for the Information
    # Bottleneck analysis.
    mu_samples = []
    logvar_samples = []
    logits_samples = []
    label_samples = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.view(
                images.size(0), -1
            ).to(device)

            labels = labels.to(device)

            logits, z, mu, logvar = model(images)

            ce = F.cross_entropy(
                logits,
                labels
            )

            info = divergence_loss(
                mu,
                logvar,
                divergence
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
            total_information += info.item()

            # Existing latent collection.
            latent_samples.append(
                z.cpu()
            )

            # New information-analysis collections.
            mu_samples.append(
                mu.cpu()
            )

            logvar_samples.append(
                logvar.cpu()
            )

            logits_samples.append(
                logits.cpu()
            )

            label_samples.append(
                labels.cpu()
            )

    accuracy = correct / total

    latent_samples = torch.cat(
        latent_samples,
        dim=0
    )

    mu_samples = torch.cat(
        mu_samples,
        dim=0
    )

    logvar_samples = torch.cat(
        logvar_samples,
        dim=0
    )

    logits_samples = torch.cat(
        logits_samples,
        dim=0
    )

    label_samples = torch.cat(
        label_samples,
        dim=0
    )

    return {
        "accuracy": accuracy,
        "cross_entropy": total_ce / len(test_loader),
        "information_loss": (
            total_information / len(test_loader)
        ),
        "latent": latent_samples,

        # Information-analysis tensors.
        "mu": mu_samples,
        "logvar": logvar_samples,
        "logits": logits_samples,
        "labels": label_samples
    }


# ---------------------------------------------------------
# Entropy estimation
# ---------------------------------------------------------

def estimate_entropy(latent):

    """
    Simple histogram-based entropy estimation.

    The latent representation is continuous, so the latent
    values are discretized into bins before applying the
    discrete entropy formulas.
    """

    values = latent.numpy().flatten()

    hist, _ = np.histogram(
        values,
        bins=100,
        density=False
    )

    probabilities = hist / hist.sum()

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
        "shannon_entropy": shannon,
        "renyi_entropy": renyi,
        "tsallis_entropy": tsallis
    }


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_bar(results, metric, title, ylabel, filename):

    methods = list(results.keys())

    values = [
        results[m][metric]
        for m in methods
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(methods, values)

    plt.xlabel("Divergence")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


def plot_accuracy_vs_information(results):

    plt.figure(figsize=(7, 5))

    for method in results:

        x = results[method][
            "information_loss"
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

    plt.xlabel("Information Loss")
    plt.ylabel("Accuracy")
    plt.title(
        "Accuracy vs Information Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "accuracy_vs_information.png"
        ),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Information-analysis plotting
# ---------------------------------------------------------

def plot_information_metrics(
    information_results
):

    information_plots = [
        (
            "I_XZ",
            "I(X;Z)",
            "Information in Latent Representation",
            "I(X;Z)",
            "I_XZ.png"
        ),
        (
            "H_Y",
            "H(Y)",
            "Entropy of Target Labels",
            "H(Y)",
            "H_Y.png"
        ),
        (
            "H_Y_given_Z",
            "H(Y|Z)",
            "Conditional Entropy of Labels Given Z",
            "H(Y|Z)",
            "H_Y_given_Z.png"
        ),
        (
            "I_ZY",
            "I(Z;Y)",
            "Information About Y Retained in Z",
            "I(Z;Y)",
            "I_ZY.png"
        ),
        (
            "IB_objective",
            "IB Objective",
            "Information Bottleneck Objective",
            "I(X;Z) - beta I(Z;Y)",
            "IB_objective.png"
        )
    ]

    for metric, title, plot_title, ylabel, filename in information_plots:

        plot_bar(
            information_results,
            metric,
            plot_title,
            ylabel,
            os.path.join(
                "..",
                "information_analysis",
                filename
            )
        )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    set_seed(SEED)

    results = {}

    information_results = {}

    histories = {}

    for divergence in DIVERGENCES:

        print("\n")
        print("=" * 60)
        print(
            f"TRAINING {divergence.upper()} VIB"
        )
        print("=" * 60)

        # Create a fresh model.
        model = VIB(
            latent_dim=LATENT_DIM
        ).to(device)

        # Train exactly as before.
        history = train_model(
            model,
            divergence
        )

        histories[divergence] = history

        # -------------------------------------------------
        # ONE evaluation pass
        #
        # This returns everything needed by BOTH:
        #   1. existing V1 evaluation/entropy
        #   2. Information Bottleneck analysis
        #
        # No second evaluation is performed.
        # -------------------------------------------------

        evaluation = evaluate_model(
            model,
            divergence
        )

        # -------------------------------------------------
        # Existing V1 entropy calculation
        # -------------------------------------------------

        entropy_results = estimate_entropy(
            evaluation["latent"]
        )

        # -------------------------------------------------
        # New Information Bottleneck analysis
        #
        # We use the exact same model outputs collected
        # during the evaluation above.
        #
        # I(X;Z) is evaluated using the KL-based expression
        # already defined in information_metrics.py.
        #
        # The training divergence is NOT changed here.
        # -------------------------------------------------

        information_metric_results = (
            calculate_all_information_metrics(
                mu=evaluation["mu"],
                logvar=evaluation["logvar"],
                labels=evaluation["labels"],
                logits=evaluation["logits"],
                num_classes=10,
                beta=BETA
            )
        )

        information_results[divergence] = (
            information_metric_results
        )

        # -------------------------------------------------
        # Existing V1 results + information metrics
        # -------------------------------------------------

        results[divergence] = {
            "accuracy":
                evaluation["accuracy"],

            "cross_entropy":
                evaluation["cross_entropy"],

            "information_loss":
                evaluation["information_loss"],

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

            # Information Bottleneck metrics.
            "I_XZ":
                information_metric_results[
                    "I_XZ"
                ],

            "H_Y":
                information_metric_results[
                    "H_Y"
                ],

            "H_Y_given_Z":
                information_metric_results[
                    "H_Y_given_Z"
                ],

            "I_ZY":
                information_metric_results[
                    "I_ZY"
                ],

            "IB_objective":
                information_metric_results[
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
            f"Information Loss: "
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

        print("\nInformation Bottleneck Metrics:")

        print(
            f"I(X;Z): "
            f"{information_metric_results['I_XZ']:.4f}"
        )

        print(
            f"H(Y): "
            f"{information_metric_results['H_Y']:.4f}"
        )

        print(
            f"H(Y|Z): "
            f"{information_metric_results['H_Y_given_Z']:.4f}"
        )

        print(
            f"I(Z;Y): "
            f"{information_metric_results['I_ZY']:.4f}"
        )

        print(
            f"IB Objective: "
            f"{information_metric_results['IB_objective']:.4f}"
        )

    # -----------------------------------------------------
    # Save ALL results to the existing V1 results.csv
    # -----------------------------------------------------

    dataframe = pd.DataFrame(
        results
    ).T

    dataframe.index.name = "divergence"

    csv_path = os.path.join(
        METRICS_DIR,
        "results.csv"
    )

    dataframe.to_csv(
        csv_path
    )

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(dataframe)

    print("\nSaved:")
    print(csv_path)

    # -----------------------------------------------------
    # Save information-only results separately
    # -----------------------------------------------------

    information_dataframe = pd.DataFrame(
        information_results
    ).T

    information_dataframe.index.name = "divergence"

    information_dataframe = information_dataframe[
        [
            "I_XZ",
            "H_Y",
            "H_Y_given_Z",
            "I_ZY",
            "IB_objective"
        ]
    ]

    information_csv_path = os.path.join(
        INFORMATION_RESULTS_DIR,
        "information_analysis.csv"
    )

    information_dataframe.to_csv(
        information_csv_path
    )

    # Readable text summary.
    information_txt_path = os.path.join(
        INFORMATION_RESULTS_DIR,
        "information_analysis.txt"
    )

    with open(
        information_txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "V1 Information Bottleneck Analysis\n"
        )
        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Beta: {BETA}\n"
        )

        f.write(
            f"Latent dimension: {LATENT_DIM}\n"
        )

        f.write(
            f"Seed: {SEED}\n\n"
        )

        f.write(
            information_dataframe.to_string()
        )

        f.write("\n")

    print("\nInformation Analysis:")
    print(information_dataframe)

    print("\nSaved:")
    print(information_csv_path)
    print(information_txt_path)

    # -----------------------------------------------------
    # Generate existing V1 graphs
    # -----------------------------------------------------

    plot_bar(
        results,
        "accuracy",
        "VIB Classification Accuracy",
        "Accuracy",
        "accuracy.png"
    )

    plot_bar(
        results,
        "cross_entropy",
        "VIB Cross Entropy",
        "Cross Entropy",
        "cross_entropy.png"
    )

    plot_bar(
        results,
        "shannon_entropy",
        "Shannon Entropy of Latent Representation",
        "Shannon Entropy",
        "shannon_entropy.png"
    )

    plot_bar(
        results,
        "renyi_entropy",
        "Renyi Entropy of Latent Representation",
        "Renyi Entropy",
        "renyi_entropy.png"
    )

    plot_bar(
        results,
        "tsallis_entropy",
        "Tsallis Entropy of Latent Representation",
        "Tsallis Entropy",
        "tsallis_entropy.png"
    )

    plot_accuracy_vs_information(
        results
    )

    # -----------------------------------------------------
    # Generate Information Bottleneck graphs
    # -----------------------------------------------------

    # The normal plot_bar function saves to PLOTS_DIR.
    # For information analysis, temporarily create the
    # required plots directly in the information-analysis
    # directory so V1 plot outputs remain separate.

    information_plot_specs = [
        (
            "I_XZ",
            "Information in Latent Representation",
            "I(X;Z)",
            "I_XZ.png"
        ),
        (
            "H_Y",
            "Entropy of Target Labels",
            "H(Y)",
            "H_Y.png"
        ),
        (
            "H_Y_given_Z",
            "Conditional Entropy of Labels Given Z",
            "H(Y|Z)",
            "H_Y_given_Z.png"
        ),
        (
            "I_ZY",
            "Information About Y Retained in Z",
            "I(Z;Y)",
            "I_ZY.png"
        ),
        (
            "IB_objective",
            "Information Bottleneck Objective",
            "IB Objective",
            "IB_objective.png"
        )
    ]

    for metric, title, ylabel, filename in information_plot_specs:

        methods = list(
            information_results.keys()
        )

        values = [
            information_results[m][metric]
            for m in methods
        ]

        plt.figure(figsize=(7, 5))

        plt.bar(
            methods,
            values
        )

        plt.xlabel("Divergence")
        plt.ylabel(ylabel)
        plt.title(title)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                INFORMATION_RESULTS_DIR,
                filename
            ),
            dpi=300
        )

        plt.close()

    print("\nGraphs saved to:")
    print(PLOTS_DIR)

    print("\nInformation-analysis graphs saved to:")
    print(INFORMATION_RESULTS_DIR)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()