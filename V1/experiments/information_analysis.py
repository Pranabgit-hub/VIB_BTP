"""
V1 Information Analysis
-----------------------

This script is intentionally separate from run_experiments.py.

It trains the same V1 VIB setup for each divergence, but computes/stores
ONLY the Information Bottleneck analysis:

    I(X;Z)      = E_X[ KL(q(z|x) || N(0,I)) ]
    H(Y)
    H(Y|Z)
    I(Z;Y)      = H(Y) - H(Y|Z)
    IB Objective = I(X;Z) - beta * I(Z;Y)

It does NOT calculate:
    - accuracy
    - cross entropy as a reported metric
    - histogram Shannon/Renyi/Tsallis entropy
    - V1 results.csv

Outputs are written only to:

    V1/results/information_analysis/

Project structure expected:

VIB_BTP/
├── data/
├── V1/
│   ├── experiments/
│   │   ├── information_metrics.py
│   │   └── information_analysis.py   <-- this file
│   ├── results/
│   └── src/
│       ├── model.py
│       ├── losses.py
│       ├── divergences.py
│       └── entropy.py
└── V2/
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# PATHS
# =========================================================

EXPERIMENTS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

V1_DIR = os.path.dirname(
    EXPERIMENTS_DIR
)

PROJECT_DIR = os.path.dirname(
    V1_DIR
)

SRC_DIR = os.path.join(
    V1_DIR,
    "src"
)

# Make V1/src importable as the "src" package.
# This also allows losses.py to use:
#     from .divergences import ...
sys.path.insert(0, V1_DIR)

# information_metrics.py is in V1/experiments
sys.path.insert(0, EXPERIMENTS_DIR)


from src.model import VIB
from src.losses import divergence_loss

from information_metrics import (
    calculate_all_information_metrics
)


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

# Same beta as V1.
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


# =========================================================
# INFORMATION ANALYSIS OUTPUT DIRECTORY
# =========================================================

INFORMATION_RESULTS_DIR = os.path.join(
    V1_DIR,
    "results",
    "information_analysis"
)

os.makedirs(
    INFORMATION_RESULTS_DIR,
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
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("V1 INFORMATION ANALYSIS")
print("=" * 65)
print(f"Device       : {device}")
print(f"Epochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Beta         : {BETA}")
print(f"Latent dim   : {LATENT_DIM}")
print("=" * 65)


# =========================================================
# DATASET
# =========================================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=os.path.join(
        PROJECT_DIR,
        "data"
    ),
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=os.path.join(
        PROJECT_DIR,
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

def train_model(model, divergence):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

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

            classification_loss = F.cross_entropy(
                logits,
                labels
            )

            # IMPORTANT:
            # Keep the original V1 divergence exactly as-is.
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

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {average_loss:.6f}"
        )


# =========================================================
# INFORMATION ANALYSIS
# =========================================================

def analyze_model(model):

    model.eval()

    mu_samples = []
    logvar_samples = []
    logits_samples = []
    label_samples = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.view(
                images.size(0),
                -1
            ).to(device)

            labels = labels.to(device)

            logits, z, mu, logvar = model(
                images
            )

            # These are the only values needed for
            # the requested information analysis.
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

    metrics = calculate_all_information_metrics(
        mu=mu_samples,
        logvar=logvar_samples,
        labels=label_samples,
        logits=logits_samples,
        num_classes=10,
        beta=BETA
    )

    return metrics


# =========================================================
# MAIN
# =========================================================

def main():

    all_results = {}

    for divergence in DIVERGENCES:

        print("\n")
        print("=" * 65)
        print(
            f"INFORMATION ANALYSIS: "
            f"{divergence.upper()}"
        )
        print("=" * 65)

        # Same V1 model architecture.
        model = VIB(
            latent_dim=LATENT_DIM
        ).to(device)

        # Same V1 training procedure.
        train_model(
            model,
            divergence
        )

        # Only information metrics are calculated here.
        metrics = analyze_model(
            model
        )

        all_results[divergence] = metrics

        print("\nInformation Metrics:")

        print(
            f"I(X;Z):       "
            f"{metrics['I_XZ']:.4f}"
        )

        print(
            f"H(Y):         "
            f"{metrics['H_Y']:.4f}"
        )

        print(
            f"H(Y|Z):       "
            f"{metrics['H_Y_given_Z']:.4f}"
        )

        print(
            f"I(Z;Y):       "
            f"{metrics['I_ZY']:.4f}"
        )

        print(
            f"IB Objective: "
            f"{metrics['IB_objective']:.4f}"
        )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame.from_dict(
        all_results,
        orient="index"
    )

    results_df.index.name = "divergence"

    results_df = results_df[
        [
            "I_XZ",
            "H_Y",
            "H_Y_given_Z",
            "I_ZY",
            "IB_objective"
        ]
    ]

    csv_path = os.path.join(
        INFORMATION_RESULTS_DIR,
        "information_analysis.csv"
    )

    results_df.to_csv(
        csv_path
    )

    # Also save a readable text summary.
    txt_path = os.path.join(
        INFORMATION_RESULTS_DIR,
        "information_analysis.txt"
    )

    with open(
        txt_path,
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
            results_df.to_string()
        )

        f.write("\n")


    print("\n")
    print("=" * 65)
    print("FINAL INFORMATION ANALYSIS")
    print("=" * 65)
    print(results_df)

    print("\nSaved:")
    print(csv_path)
    print(txt_path)

    print("\nExisting V1 results were NOT modified.")
    print("No histogram entropy was calculated by this script.")


if __name__ == "__main__":
    main()