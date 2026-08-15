import torch


def calculate_i_xz(mu, logvar):
    
    var = torch.exp(logvar)

    kl_per_sample = 0.5 * (
        mu.pow(2)
        + var
        - 1.0
        - logvar
    ).sum(dim=-1)

    return kl_per_sample.mean().item()


def calculate_h_y(labels, num_classes):

    counts = torch.bincount(
        labels,
        minlength=num_classes
    ).float()

    probabilities = counts / counts.sum()

    probabilities = probabilities[
        probabilities > 0
    ]

    h_y = -(
        probabilities
        * torch.log(probabilities)
    ).sum()

    return h_y.item()


def calculate_h_y_given_z(logits):

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    log_probabilities = torch.log(
        probabilities.clamp_min(1e-12)
    )

    entropy_per_sample = -(
        probabilities
        * log_probabilities
    ).sum(dim=-1)

    return entropy_per_sample.mean().item()


def calculate_i_zy(
    labels,
    logits,
    num_classes
):
    """
    I(Z;Y) = H(Y) - H(Y|Z)
    """

    h_y = calculate_h_y(
        labels,
        num_classes
    )

    h_y_given_z = calculate_h_y_given_z(
        logits
    )

    i_zy = h_y - h_y_given_z

    return {
        "H_Y": h_y,
        "H_Y_given_Z": h_y_given_z,
        "I_ZY": i_zy
    }


def calculate_ib_objective(
    i_xz,
    i_zy,
    beta
):
    """
    Information Bottleneck objective:

        I(X;Z) - beta * I(Z;Y)
    """

    return i_xz - beta * i_zy


def calculate_all_information_metrics(
    mu,
    logvar,
    labels,
    logits,
    num_classes,
    beta
):
    """
    Calculate all information-theoretic evaluation metrics
    for one trained model.
    """

    # Common KL-based I(X;Z) evaluation
    i_xz = calculate_i_xz(
        mu,
        logvar
    )

    # H(Y), H(Y|Z), I(Z;Y)
    zy_metrics = calculate_i_zy(
        labels,
        logits,
        num_classes
    )

    h_y = zy_metrics["H_Y"]
    h_y_given_z = zy_metrics["H_Y_given_Z"]
    i_zy = zy_metrics["I_ZY"]

    # IB objective
    ib_objective = calculate_ib_objective(
        i_xz,
        i_zy,
        beta
    )

    return {
        "I_XZ": i_xz,
        "H_Y": h_y,
        "H_Y_given_Z": h_y_given_z,
        "I_ZY": i_zy,
        "IB_objective": ib_objective
    }