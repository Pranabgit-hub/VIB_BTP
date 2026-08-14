import torch


def kl_divergence_gaussian(mu, logvar):
    """
    KL(q(z|x) || N(0,I))

    q(z|x) = N(mu, diag(exp(logvar)))
    """

    kl = 0.5 * (
        torch.exp(logvar)
        + mu ** 2
        - 1
        - logvar
    )

    return kl.sum(dim=1).mean()


def renyi_divergence_gaussian(mu, logvar, alpha=2.0):
    """
    Rényi divergence between:
        q = N(mu, diag(exp(logvar)))
        p = N(0,I)

    Numerical Monte-Carlo approximation.
    """

    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and != 1")

    std = torch.exp(0.5 * logvar)

    eps = torch.randn_like(std)

    z = mu + std * eps

    log_q = (
        -0.5 * (
            ((z - mu) / std) ** 2
            + logvar
            + torch.log(torch.tensor(2.0 * torch.pi,
                                     device=z.device))
        )
    ).sum(dim=1)

    log_p = (
        -0.5 * (
            z ** 2
            + torch.log(torch.tensor(2.0 * torch.pi,
                                     device=z.device))
        )
    ).sum(dim=1)

    log_integrand = alpha * log_q + (1 - alpha) * log_p

    divergence = torch.logsumexp(
        log_integrand, dim=0
    ) - torch.log(
        torch.tensor(float(z.shape[0]), device=z.device)
    )

    return divergence / (alpha - 1)


def tsallis_divergence_gaussian(mu, logvar, alpha=2.0):
    """
    Tsallis divergence using the same Monte-Carlo
    integral used for Rényi divergence.
    """

    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and != 1")

    std = torch.exp(0.5 * logvar)

    eps = torch.randn_like(std)

    z = mu + std * eps

    log_q = (
        -0.5 * (
            ((z - mu) / std) ** 2
            + logvar
            + torch.log(torch.tensor(2.0 * torch.pi,
                                     device=z.device))
        )
    ).sum(dim=1)

    log_p = (
        -0.5 * (
            z ** 2
            + torch.log(torch.tensor(2.0 * torch.pi,
                                     device=z.device))
        )
    ).sum(dim=1)

    ratio_term = torch.exp(
        (alpha - 1) * (log_q - log_p)
    )

    divergence = (
        ratio_term.mean() - 1
    ) / (alpha - 1)

    return divergence