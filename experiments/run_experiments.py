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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.model import VIB
from src.losses import divergence_loss
from src.entropy import (
    shannon_entropy,
    renyi_entropy,
    tsallis_entropy
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

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


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

def evaluate_model(model):

    model.eval()

    correct = 0
    total = 0

    total_ce = 0.0
    total_information = 0.0

    latent_samples = []

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
                "kl"
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

            latent_samples.append(
                z.cpu()
            )

    accuracy = correct / total

    latent_samples = torch.cat(
        latent_samples,
        dim=0
    )

    return {
        "accuracy": accuracy,
        "cross_entropy": total_ce / len(test_loader),
        "information_loss": (
            total_information / len(test_loader)
        ),
        "latent": latent_samples
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

        # Create a fresh model
        model = VIB(
            latent_dim=LATENT_DIM
        ).to(device)

        # Train
        history = train_model(
            model,
            divergence
        )

        histories[divergence] = history

        # Evaluate
        evaluation = evaluate_model(
            model
        )

        # Entropy
        entropy_results = estimate_entropy(
            evaluation["latent"]
        )

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
                ]
        }

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

    # -----------------------------------------------------
    # Save results
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
    # Generate graphs
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

    print("\nGraphs saved to:")
    print(PLOTS_DIR)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()