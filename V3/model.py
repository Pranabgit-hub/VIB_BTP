import torch
import torch.nn as nn


class VIB(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        latent_dim=32,
        num_classes=10
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    # =====================================================
    # ENCODER
    # =====================================================

    def encode(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)

        raw_logvar = self.fc_logvar(h)

        # -------------------------------------------------
        # Numerical stability
        #
        # Prevent:
        #
        #   exp(logvar)
        #
        # from becoming numerically extreme.
        #
        # This does NOT change the Gaussian formulation.
        # It simply restricts the learned covariance to a
        # numerically reasonable positive range.
        # -------------------------------------------------

        logvar = torch.clamp(
            raw_logvar,
            min=-8.0,
            max=8.0
        )

        return mu, logvar

    # =====================================================
    # REPARAMETERIZATION
    # =====================================================

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return mu + eps * std

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        logits = self.classifier(
            z
        )

        return (
            logits,
            z,
            mu,
            logvar
        )