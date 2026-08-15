import torch


def kl_gaussian(mu, logvar):
    """
    KL(q(z|x) || p(z))

    q(z|x) = N(mu, diag(exp(logvar)))
    p(z)   = N(0, I)

    Returns mean KL over the batch.
    """
    var = torch.exp(logvar)

    kl = 0.5 * (
        mu.pow(2)
        + var
        - 1.0
        - logvar
    )

    return kl.sum(dim=-1).mean()


def gaussian_log_integral_power(mu, logvar, alpha):
    """
    Exact multivariate Gaussian calculation of

        log A_alpha

    where

        A_alpha = integral q(z)^alpha p(z)^(1-alpha) dz

    q(z) = N(mu, Sigma_q)
    p(z) = N(0, I)

    Sigma_q is diagonal.

    No Monte Carlo approximation is used.
    """

    if alpha <= 0:
        raise ValueError("alpha must be > 0.")

    if abs(alpha - 1.0) < 1e-8:
        raise ValueError(
            "alpha cannot be exactly 1 for Renyi/Tsallis."
        )

    var = torch.exp(logvar)

    # Sigma_q^{-1}
    precision_q = 1.0 / var

    # A = alpha Sigma_q^{-1} + (1-alpha) I
    A = alpha * precision_q + (1.0 - alpha)

    # Integral must be finite
    if torch.any(A <= 0):
        raise ValueError(
            "Renyi/Tsallis Gaussian integral is not finite "
            "for the current alpha and covariance."
        )

    # log |Sigma_q|
    logdet_sigma_q = logvar.sum(dim=-1)

    # log |A|
    logdet_A = torch.log(A).sum(dim=-1)

    # Mean-dependent term
    mean_term = (
        -0.5
        * alpha
        * (1.0 - alpha)
        * (
            mu.pow(2)
            * precision_q
            / A
        )
    ).sum(dim=-1)

    log_A = (
        -0.5 * alpha * logdet_sigma_q
        -0.5 * logdet_A
        + mean_term
    )

    return log_A


def renyi_gaussian(mu, logvar, alpha=2.0):
    """
    Exact Renyi divergence:

        D_R^alpha(q || p)
        = log(A_alpha) / (alpha - 1)
    """

    if abs(alpha - 1.0) < 1e-8:
        return kl_gaussian(mu, logvar)

    log_A = gaussian_log_integral_power(
        mu,
        logvar,
        alpha
    )

    divergence = log_A / (alpha - 1.0)

    return divergence.mean()


def tsallis_gaussian(mu, logvar, alpha=2.0):
    """
    Exact Tsallis divergence:

        D_T^alpha(q || p)
        = (A_alpha - 1) / (alpha - 1)
    """

    if abs(alpha - 1.0) < 1e-8:
        return kl_gaussian(mu, logvar)

    log_A = gaussian_log_integral_power(
        mu,
        logvar,
        alpha
    )

    A = torch.exp(log_A)

    divergence = (
        A - 1.0
    ) / (alpha - 1.0)

    return divergence.mean()


def divergence_loss(
    mu,
    logvar,
    divergence="kl",
    alpha=2.0
):
    """
    Select the divergence used for VIB training.
    """

    if divergence == "kl":
        return kl_gaussian(mu, logvar)

    if divergence == "renyi":
        return renyi_gaussian(
            mu,
            logvar,
            alpha
        )

    if divergence == "tsallis":
        return tsallis_gaussian(
            mu,
            logvar,
            alpha
        )

    raise ValueError(
        f"Unknown divergence: {divergence}"
    )