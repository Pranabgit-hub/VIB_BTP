import torch


def classification_accuracy(logits, labels):

    predictions = torch.argmax(
        logits, dim=1
    )

    return (
        predictions == labels
    ).float().mean().item()


def average_cross_entropy(
    logits,
    labels
):

    return torch.nn.functional.cross_entropy(
        logits,
        labels
    ).item()