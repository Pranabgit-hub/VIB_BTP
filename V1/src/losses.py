import torch.nn.functional as F

from .divergences import (
    kl_divergence_gaussian,
    renyi_divergence_gaussian,
    tsallis_divergence_gaussian
)


def divergence_loss(mu, logvar, method="kl"):

    if method == "kl":
        return kl_divergence_gaussian(
            mu, logvar
        )

    elif method == "renyi":
        return renyi_divergence_gaussian(
            mu, logvar
        )

    elif method == "tsallis":
        return tsallis_divergence_gaussian(
            mu, logvar
        )

    else:
        raise ValueError(
            f"Unknown divergence: {method}"
        )


def vib_loss(
    logits,
    labels,
    mu,
    logvar,
    beta,
    divergence="kl"
):

    classification_loss = F.cross_entropy(
        logits,
        labels
    )

    information_loss = divergence_loss(
        mu,
        logvar,
        divergence
    )

    total_loss = (
        classification_loss
        + beta * information_loss
    )

    return (
        total_loss,
        classification_loss,
        information_loss
    )