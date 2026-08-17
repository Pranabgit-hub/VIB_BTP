import torch


# =========================================================
# KL DIVERGENCE
# =========================================================

def kl_gaussian(mu, logvar):
    """
    Exact Gaussian KL divergence:

        KL(q(z|x) || p(z))

    where

        q(z|x) = N(mu, diag(exp(logvar)))
        p(z)   = N(0, I)

    This is the standard VIB KL term.

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
    Exact Gaussian calculation of:

        log A_alpha

    where

        A_alpha =
            integral q(z)^alpha p(z)^(1-alpha) dz

    q(z) = N(mu, Sigma_q)
    p(z) = N(0, I)

    Sigma_q is diagonal.

    -------------------------------------------------------
    IMPORTANT
    -------------------------------------------------------

    We intentionally use:

        0 < alpha < 1

    For this range the Gaussian power integral is finite
    for every positive-definite covariance matrix.

    This avoids the problem encountered with:

        alpha = 2

    where the integral requires an additional condition:

        alpha / sigma_i^2 + (1-alpha) > 0

    and the learned covariance could violate this condition.

    No Monte Carlo approximation is used.
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "Gaussian Renyi/Tsallis training requires "
            "0 < alpha < 1."
        )

    # -----------------------------------------------------
    # Constants
    # -----------------------------------------------------

    alpha_t = torch.as_tensor(
        alpha,
        dtype=mu.dtype,
        device=mu.device
    )

    one_minus_alpha_t = torch.as_tensor(
        1.0 - alpha,
        dtype=mu.dtype,
        device=mu.device
    )

    # -----------------------------------------------------
    # log determinant of Sigma_q
    #
    # Sigma_q = diag(exp(logvar))
    #
    # therefore:
    #
    # log |Sigma_q| = sum(logvar)
    # -----------------------------------------------------

    logdet_sigma_q = logvar.sum(dim=-1)

    # -----------------------------------------------------
    # A = alpha Sigma^{-1} + (1-alpha)I
    #
    # Instead of computing:
    #
    #     1 / exp(logvar)
    #
    # directly, calculate log(A) using logaddexp.
    #
    # A_i =
    #
    #     alpha / variance_i + (1-alpha)
    #
    # -----------------------------------------------------

    log_alpha = torch.log(
        alpha_t
    )

    log_one_minus_alpha = torch.log(
        one_minus_alpha_t
    )

    log_A_diagonal = torch.logaddexp(
        log_alpha - logvar,
        log_one_minus_alpha
    )

    logdet_A = log_A_diagonal.sum(dim=-1)

    # -----------------------------------------------------
    # Mean-dependent term
    #
    # The exact expression simplifies to:
    #
    # -1/2 * alpha * (1-alpha)
    #
    # * mu_i^2
    # -----------------------------
    #   alpha + (1-alpha)*variance_i
    #
    # -----------------------------------------------------

    # denominator:
    #
    # alpha + (1-alpha) exp(logvar)

    log_denominator = torch.logaddexp(
        log_alpha,
        log_one_minus_alpha + logvar
    )

    denominator = torch.exp(
        log_denominator
    )

    mean_term = (
        -0.5
        * alpha_t
        * one_minus_alpha_t
        * mu.pow(2)
        / denominator
    ).sum(dim=-1)

    # -----------------------------------------------------
    # Exact log integral
    #
    # log A_alpha =
    #
    # -(alpha/2) log|Sigma|
    #
    # -(1/2) log|A|
    #
    # + mean-dependent term
    # -----------------------------------------------------

    log_integral = (
        -0.5
        * alpha_t
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
    Exact Gaussian Renyi divergence:

        D_R^alpha(q || p)

        =
        log(A_alpha)
        ----------------
          alpha - 1

    where:

        A_alpha =
            integral q(z)^alpha p(z)^(1-alpha) dz

    No Monte Carlo approximation.

    -------------------------------------------------------
    V3 choice
    -------------------------------------------------------

        alpha = 0.5

    This is deliberately chosen because:

        0 < alpha < 1

    guarantees a finite Gaussian power integral for any
    positive covariance.

    For alpha = 0.5:

        D_R^0.5(q || p)
        =
        -2 log(A_0.5)

    -------------------------------------------------------
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "V3 Renyi training requires 0 < alpha < 1."
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
    Exact Gaussian Tsallis divergence:

        D_T^alpha(q || p)

        =
        (A_alpha - 1)
        ----------------
           alpha - 1

    where:

        A_alpha =
            integral q(z)^alpha p(z)^(1-alpha) dz

    No Monte Carlo approximation.

    -------------------------------------------------------
    V3 choice
    -------------------------------------------------------

        alpha = 0.5

    Then:

        D_T^0.5
        =
        2(1 - A_0.5)

    Since A_0.5 is positive:

        D_T^0.5 < 2

    This makes the Tsallis training objective naturally
    bounded and avoids the extremely large values seen
    with the previous alpha=2 formulation.

    expm1() is used for numerical stability.
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            "V3 Tsallis training requires 0 < alpha < 1."
        )

    log_integral = gaussian_log_integral_power(
        mu,
        logvar,
        alpha
    )

    # exp(log_integral) - 1
    #
    # expm1 is more accurate when log_integral is close
    # to zero.

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
    Select the divergence used during V3 training.

    KL:
        Exact Gaussian KL.

    Renyi:
        Exact Gaussian Renyi with alpha=0.5.

    Tsallis:
        Exact Gaussian Tsallis with alpha=0.5.
    """

    if divergence == "kl":

        return kl_gaussian(
            mu,
            logvar
        )

    elif divergence == "renyi":

        return renyi_gaussian(
            mu,
            logvar,
            alpha
        )

    elif divergence == "tsallis":

        return tsallis_gaussian(
            mu,
            logvar,
            alpha
        )

    else:

        raise ValueError(
            f"Unknown divergence: {divergence}"
        )