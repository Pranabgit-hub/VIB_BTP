import torch
import torch.nn as nn


# =========================================================
# SHARED VIB COMPONENTS
# =========================================================

def clamp_logvar(raw_logvar):
    """
    Numerical stability.

    Prevent:

        exp(logvar)

    from becoming numerically extreme.

    This does NOT change the Gaussian formulation.
    It simply restricts the learned covariance to a
    numerically reasonable positive range.
    """

    return torch.clamp(
        raw_logvar,
        min=-8.0,
        max=8.0
    )


class VIBBase(nn.Module):
    """
    Shared reparameterization + forward logic.

    Subclasses must define:

        self.encode(x)      -> (mu, logvar)
        self.classifier     -> nn.Module mapping z -> logits
    """

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


# =========================================================
# MLP VARIANT  (flat vector inputs, e.g. MNIST)
# =========================================================

class VIBMLP(VIBBase):

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

    def encode(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)

        logvar = clamp_logvar(
            self.fc_logvar(h)
        )

        return mu, logvar


# Backwards-compatible alias (original class name).
VIB = VIBMLP


# =========================================================
# CNN VARIANT  (image tensor inputs, e.g. CIFAR-10)
#
# Works for any input size >= ~16 px per side:
#   - two /2 max pools shrink H, W to H/4, W/4
#   - AdaptiveAvgPool2d((4, 4)) normalises the result
#     to a fixed 4x4 grid regardless of input size.
#
# Therefore the same architecture serves 28x28 MNIST,
# 32x32 CIFAR-10, or any other image dataset without
# computing feature dims by hand.
# =========================================================

class VIBCNN(VIBBase):

    def __init__(
        self,
        input_shape=(3, 32, 32),
        hidden_dim=256,
        latent_dim=64,
        num_classes=10
    ):

        super().__init__()

        in_channels = input_shape[0]

        self.features = nn.Sequential(

            # ---- Block 1: C x H x W -> 32 x H/2 x W/2 ----
            nn.Conv2d(
                in_channels, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # ---- Block 2: -> 64 x H/4 x W/4 ----
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # ---- Block 3: -> fixed 128 x 4 x 4 ----
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        feature_dim = 128 * 4 * 4

        self.fc_hidden = nn.Sequential(

            nn.Linear(
                feature_dim,
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

    def encode(self, x):

        f = self.features(x)

        f = f.view(f.size(0), -1)

        h = self.fc_hidden(f)

        mu = self.fc_mu(h)

        logvar = clamp_logvar(
            self.fc_logvar(h)
        )

        return mu, logvar


# =========================================================
# ARCHITECTURE REGISTRY
#
# run_analysis.py selects the architecture via the
# dataset config ("arch" key). To support a new
# architecture add an entry here — no other file needs
# to change.
# =========================================================

ARCH_REGISTRY = {
    "mlp": VIBMLP,
    "cnn": VIBCNN,
}


def build_vib(
    arch="mlp",
    input_shape=(1, 28, 28),
    hidden_dim=256,
    latent_dim=32,
    num_classes=10
):
    """
    Build a VIB model for the given architecture.

    Both variants expose the identical interface used
    everywhere in this analysis:

        model.encode(x)          -> (mu, logvar)
        model.classifier(z)      -> logits
        model(x)                 -> (logits, z, mu, logvar)

    The MLP consumes flattened vectors of size
    prod(input_shape); the CNN consumes image tensors of
    shape (B, C, H, W).
    """

    if arch not in ARCH_REGISTRY:

        raise ValueError(
            f"Unknown architecture: {arch}. "
            f"Choose from: {list(ARCH_REGISTRY.keys())}"
        )

    arch_cls = ARCH_REGISTRY[arch]

    if arch == "cnn":

        return arch_cls(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=num_classes
        )

    input_dim = 1

    for dim in input_shape:

        input_dim *= dim

    return arch_cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_classes=num_classes
    )
