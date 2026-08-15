import torch


# =========================================================
# KL DIVERGENCE
# =========================================================

def kl_gaussian(mu, logvar):
    """
    KL(q(z|x) || p(z))

    q(z|x) = N(mu, diag(exp(logvar)))
    p(z)   = N(0, I)

    Exact Gaussian KL divergence.

    Returns:
        Mean KL over the batch.
    """

    var = torch.exp(logvar)

    kl = 0.5 * (
        mu.pow(2)
        + var
        - 1.0
        - logvar
    )

    return kl.sum(dim=-1).mean()


# =========================================================
# EXACT GAUSSIAN POWER INTEGRAL
# =========================================================

def gaussian_log_integral_power(
    mu,
    logvar,
    alpha
):
    """
    Exact Gaussian calculation of

        log A_alpha

    where

        A_alpha
        = integral q(z)^alpha p(z)^(1-alpha) dz

    q(z) = N(mu, Sigma_q)
    p(z) = N(0, I)

    Sigma_q is diagonal.

    IMPORTANT
    ---------
    We intentionally use:

        0 < alpha < 1

    because for alpha in this range the Gaussian power
    integral is finite for every positive-definite covariance
    Sigma_q.

    Therefore the training cannot fail because of the
    condition:

        alpha Sigma_q^{-1}
        + (1-alpha)I > 0

    which caused the alpha=2 formulation to go out of
    bounds.

    No Monte Carlo approximation is used.
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "For stable Gaussian Renyi/Tsallis training, "
            "alpha must satisfy 0 < alpha < 1."
        )

    # -----------------------------------------------------
    # Stable computation
    #
    # We avoid explicitly computing Sigma^{-1} where
    # possible and use logaddexp for numerical stability.
    # -----------------------------------------------------

    log_alpha = torch.log(
        torch.tensor(
            alpha,
            dtype=mu.dtype,
            device=mu.device
        )
    )

    log_one_minus_alpha = torch.log(
        torch.tensor(
            1.0 - alpha,
            dtype=mu.dtype,
            device=mu.device
        )
    )

    # -----------------------------------------------------
    # log |Sigma_q|
    #
    # Sigma_q is diagonal:
    #
    # log |Sigma_q| = sum(logvar)
    # -----------------------------------------------------

    logdet_sigma_q = logvar.sum(dim=-1)

    # -----------------------------------------------------
    # A = alpha Sigma_q^{-1}
    #     + (1-alpha)I
    #
    # For each latent dimension:
    #
    # A_i = alpha / variance_i + (1-alpha)
    #
    # Instead of explicitly calculating 1/variance,
    # compute:
    #
    # log(A_i)
    # = logaddexp(
    #       log(alpha) - logvar_i,
    #       log(1-alpha)
    #   )
    # -----------------------------------------------------

    log_A_matrix = torch.logaddexp(
        log_alpha - logvar,
        log_one_minus_alpha
    )

    logdet_A = log_A_matrix.sum(dim=-1)

    # -----------------------------------------------------
    # Mean-dependent term
    #
    # The exact expression simplifies to:
    #
    # -1/2 * alpha * (1-alpha)
    #
    # * mu_i^2
    # -------------------------------
    #   alpha + (1-alpha) variance_i
    #
    # We compute the denominator in log-space.
    # -----------------------------------------------------

    log_denominator = torch.logaddexp(
        log_alpha,
        log_one_minus_alpha + logvar
    )

    denominator = torch.exp(
        log_denominator
    )

    mean_term = (
        -0.5
        * alpha
        * (1.0 - alpha)
        * mu.pow(2)
        / denominator
    ).sum(dim=-1)

    # -----------------------------------------------------
    # Exact log integral
    #
    # log A_alpha =
    #
    # -alpha/2 log|Sigma_q|
    # -1/2 log|A|
    # + mean term
    # -----------------------------------------------------

    log_integral = (
        -0.5
        * alpha
        * logdet_sigma_q

        -0.5
        * logdet_A

        + mean_term
    )

    return log_integral


# =========================================================
# RENYI DIVERGENCE
# =========================================================

def renyi_gaussian(
    mu,
    logvar,
    alpha=0.5
):
    """
    Exact Gaussian Renyi divergence.

        D_R^alpha(q || p)
        =
        1 / (alpha - 1)
        * log(
            integral q(z)^alpha p(z)^(1-alpha) dz
        )

    No Monte Carlo approximation.

    For V2 we use alpha=0.5.

    alpha=0.5 is particularly stable because:

        0 < alpha < 1

    guarantees that the Gaussian power integral is finite
    for any positive covariance.
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "V2 Renyi training requires 0 < alpha < 1."
        )

    log_integral = gaussian_log_integral_power(
        mu,
        logvar,
        alpha
    )

    divergence = (
        log_integral
        / (alpha - 1.0)
    )

    return divergence.mean()


# =========================================================
# TSALLIS DIVERGENCE
# =========================================================

def tsallis_gaussian(
    mu,
    logvar,
    alpha=0.5
):
    """
    Exact Gaussian Tsallis divergence.

        D_T^alpha(q || p)
        =
        [A_alpha - 1]
        / (alpha - 1)

    where

        A_alpha
        =
        integral q(z)^alpha p(z)^(1-alpha) dz

    No Monte Carlo approximation.

    We use expm1(log_integral) instead of:

        exp(log_integral) - 1

    for better numerical stability.
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "V2 Tsallis training requires 0 < alpha < 1."
        )

    log_integral = gaussian_log_integral_power(
        mu,
        logvar,
        alpha
    )

    # exp(log_integral) - 1
    # computed stably
    integral_minus_one = torch.expm1(
        log_integral
    )

    divergence = (
        integral_minus_one
        / (alpha - 1.0)
    )

    return divergence.mean()


# =========================================================
# DIVERGENCE SELECTOR
# =========================================================

def divergence_loss(
    mu,
    logvar,
    divergence="kl",
    alpha=0.5
):
    """
    Select the divergence used during VIB training.

    KL:
        Exact Gaussian KL.

    Renyi:
        Exact Gaussian Renyi divergence.

    Tsallis:
        Exact Gaussian Tsallis divergence.
    """

    if divergence == "kl":

        return kl_gaussian(
            mu,
            logvar
        )

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