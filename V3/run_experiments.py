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

V3_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    V3_DIR
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


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 128

EPOCHS = 20

LEARNING_RATE = 1e-3

BETA = 1e-3

LATENT_DIM = 32


# ---------------------------------------------------------
# IMPORTANT:
#
# V3 uses alpha = 0.5 for both divergences.
#
# This is NOT Monte Carlo.
#
# Both are calculated using the exact Gaussian closed form.
# ---------------------------------------------------------

RENYI_ALPHA = 0.95

TSALLIS_ALPHA = 0.95


SEED = 42


DIVERGENCES = [
    "kl",
    "renyi",
    "tsallis"
]


# =========================================================
# RESULTS DIRECTORIES
# =========================================================

RESULTS_DIR = os.path.join(
    V3_DIR,
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

INFORMATION_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "information_analysis"
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
    "VIB VERSION 3 - EXACT GAUSSIAN DIVERGENCES"
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


            # -------------------------------------------------
            # Classification loss
            # -------------------------------------------------

            classification_loss = F.cross_entropy(
                logits,
                labels
            )


            # -------------------------------------------------
            # Training divergence
            #
            # KL:
            #   Gaussian KL
            #
            # Renyi:
            #   Exact Gaussian Renyi,
            #   alpha = 0.5
            #
            # Tsallis:
            #   Exact Gaussian Tsallis,
            #   alpha = 0.5
            # -------------------------------------------------

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


            # -------------------------------------------------
            # VIB objective
            # -------------------------------------------------

            loss = (
                classification_loss
                + BETA * information_loss
            )


            # -------------------------------------------------
            # Numerical sanity check
            # -------------------------------------------------

            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss encountered during "
                    f"{divergence} training."
                )


            loss.backward()


            # -------------------------------------------------
            # Gradient clipping
            #
            # This does NOT change the divergence formula.
            #
            # It prevents one unstable gradient step from
            # destroying the learned Gaussian parameters.
            # -------------------------------------------------

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
# EVALUATION
# =========================================================

def evaluate_model(
    model,
    divergence
):

    model.eval()


    correct = 0

    total = 0


    total_ce = 0.0

    total_information = 0.0


    latent_samples = []

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


            logits, z, mu, logvar = model(
                images
            )


            ce = F.cross_entropy(
                logits,
                labels
            )


            if divergence == "renyi":

                alpha = RENYI_ALPHA

            elif divergence == "tsallis":

                alpha = TSALLIS_ALPHA

            else:

                alpha = 0.5


            info = divergence_loss(
                mu,
                logvar,
                divergence=divergence,
                alpha=alpha
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
        correct
        / total
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
            total_ce
            / len(test_loader),

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


# =========================================================
# ENTROPY ESTIMATION
# =========================================================

def estimate_entropy(
    latent
):

    values = (
        latent.numpy()
        .flatten()
    )


    hist, _ = np.histogram(
        values,
        bins=100,
        density=False
    )


    probabilities = (
        hist
        / hist.sum()
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


# =========================================================
# PLOTTING
# =========================================================

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


    plt.xlabel(
        "Training Divergence"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Accuracy vs Training Divergence"
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


def plot_ib_tradeoff(
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


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(
        SEED
    )


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
        # Entropy
        # -------------------------------------------------

        entropy_results = estimate_entropy(
            evaluation["latent"]
        )


        # -------------------------------------------------
        # INFORMATION BOTTLENECK ANALYSIS
        #
        # IMPORTANT:
        #
        # I(X;Z) is ALWAYS calculated using the
        # KL Gaussian upper-bound expression.
        #
        # We DO NOT calculate I(X;Z) using the
        # Renyi or Tsallis training divergence.
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


        information_results[
            divergence
        ] = information_metric_results


        # -------------------------------------------------
        # Combine results
        # -------------------------------------------------

        results[
            divergence
        ] = {

            "accuracy":
                evaluation[
                    "accuracy"
                ],

            "cross_entropy":
                evaluation[
                    "cross_entropy"
                ],

            "information_loss":
                evaluation[
                    "information_loss"
                ],

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

            # ---------------------------------------------
            # Common KL-based IB analysis
            # ---------------------------------------------

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
            f"Training {divergence.upper()} "
            f"Divergence: "
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
            "\nInformation Bottleneck Metrics:"
        )


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


    # =====================================================
    # SAVE MAIN RESULTS
    # =====================================================

    dataframe = pd.DataFrame(
        results
    ).T


    dataframe.index.name = (
        "divergence"
    )


    csv_path = os.path.join(
        METRICS_DIR,
        "results_v3.csv"
    )


    dataframe.to_csv(
        csv_path
    )


    print("\n")

    print("=" * 60)

    print(
        "V3 FINAL RESULTS"
    )

    print("=" * 60)

    print(
        dataframe
    )


    print("\nSaved:")

    print(
        csv_path
    )


    # =====================================================
    # SAVE INFORMATION ANALYSIS
    # =====================================================

    information_dataframe = pd.DataFrame(
        information_results
    ).T


    information_dataframe.index.name = (
        "divergence"
    )


    information_dataframe = (
        information_dataframe[
            [
                "I_XZ",
                "H_Y",
                "H_Y_given_Z",
                "I_ZY",
                "IB_objective"
            ]
        ]
    )


    information_csv_path = os.path.join(
        INFORMATION_RESULTS_DIR,
        "information_analysis.csv"
    )


    information_dataframe.to_csv(
        information_csv_path
    )


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
            "V3 Information Bottleneck Analysis\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Beta: {BETA}\n"
        )

        f.write(
            f"Latent dimension: {LATENT_DIM}\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Renyi alpha: {RENYI_ALPHA}\n"
        )

        f.write(
            f"Tsallis alpha: {TSALLIS_ALPHA}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "I(X;Z) is evaluated using the "
            "Gaussian KL upper bound for all models.\n"
        )

        f.write(
            "The Renyi/Tsallis divergences are used "
            "only for training.\n\n"
        )

        f.write(
            information_dataframe.to_string()
        )

        f.write(
            "\n"
        )


    print(
        "\nInformation Analysis:"
    )

    print(
        information_dataframe
    )


    print("\nSaved:")

    print(
        information_csv_path
    )

    print(
        information_txt_path
    )


    # =====================================================
    # PLOTS
    # =====================================================

    plot_bar(
        results,
        "accuracy",
        "V3 VIB Classification Accuracy",
        "Accuracy",
        "accuracy_v3.png"
    )


    plot_bar(
        results,
        "cross_entropy",
        "V3 VIB Cross Entropy",
        "Cross Entropy",
        "cross_entropy_v3.png"
    )


    plot_bar(
        results,
        "information_loss",
        "V3 Training Divergence",
        "Divergence",
        "training_divergence_v3.png"
    )


    plot_bar(
        results,
        "shannon_entropy",
        "Shannon Entropy of Latent Representation",
        "Shannon Entropy",
        "shannon_entropy_v3.png"
    )


    plot_bar(
        results,
        "renyi_entropy",
        "Renyi Entropy of Latent Representation",
        "Renyi Entropy",
        "renyi_entropy_v3.png"
    )


    plot_bar(
        results,
        "tsallis_entropy",
        "Tsallis Entropy of Latent Representation",
        "Tsallis Entropy",
        "tsallis_entropy_v3.png"
    )


    plot_bar(
        results,
        "I_XZ",
        "I(X;Z) - KL Gaussian Upper Bound",
        "I(X;Z)",
        "I_XZ_v3.png"
    )


    plot_bar(
        results,
        "I_ZY",
        "I(Z;Y)",
        "I(Z;Y)",
        "I_ZY_v3.png"
    )


    plot_bar(
        results,
        "IB_objective",
        "Information Bottleneck Objective",
        "I(X;Z) - Beta I(Z;Y)",
        "IB_objective_v3.png"
    )


    plot_accuracy_vs_information(
        results
    )


    plot_ib_tradeoff(
        results
    )


    # =====================================================
    # INFORMATION ANALYSIS PLOTS
    # =====================================================

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


    for (
        metric,
        title,
        ylabel,
        filename
    ) in information_plot_specs:

        methods = list(
            information_results.keys()
        )


        values = [
            information_results[m][metric]
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
                INFORMATION_RESULTS_DIR,
                filename
            ),
            dpi=300
        )


        plt.close()


    print(
        "\nGraphs saved to:"
    )

    print(
        PLOTS_DIR
    )


    print(
        "\nInformation-analysis graphs saved to:"
    )

    print(
        INFORMATION_RESULTS_DIR
    )


    print(
        "\nExperiment completed."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()