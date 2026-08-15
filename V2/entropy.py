import torch


def shannon_entropy(probabilities):

    probabilities = probabilities[
        probabilities > 0
    ]

    return -torch.sum(
        probabilities * torch.log(probabilities)
    )


def renyi_entropy(probabilities, alpha=2.0):

    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and != 1")

    probabilities = probabilities[
        probabilities > 0
    ]

    return (
        1 / (1 - alpha)
    ) * torch.log(
        torch.sum(probabilities ** alpha)
    )


def tsallis_entropy(probabilities, alpha=2.0):

    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and != 1")

    probabilities = probabilities[
        probabilities > 0
    ]

    return (
        1 / (alpha - 1)
    ) * (
        1 - torch.sum(probabilities ** alpha)
    )