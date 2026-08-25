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
# 1D ResNet VARIANT  (time-series inputs, e.g. FordA)
#
# ResNet-18-style architecture adapted for 1-D signals:
#   Initial Conv1d -> 4 stages of residual blocks
#   -> Global Average Pooling -> VIB bottleneck
#
# Input: (B, in_channels, seq_len)
#   e.g. FordA: (B, 1, 500)
# =========================================================


class _ResBlock1D(nn.Module):
    """Basic 1-D residual block: two Conv1d + BN + ReLU
    with a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class VIBResNet1D(VIBBase):
    """
    ResNet-18-style encoder for 1-D time-series,
    wrapped with VIB bottleneck (mu / logvar / z).

    Architecture:
        Initial conv:  in_channels -> 64
        Stage 1:  2 x ResBlock(64,  stride=1)
        Stage 2:  2 x ResBlock(128, stride=2)
        Stage 3:  2 x ResBlock(256, stride=2)
        Stage 4:  2 x ResBlock(512, stride=2)
        Global average pool over time -> 512-d
        -> fc_hidden -> mu / logvar -> classifier
    """

    def __init__(
        self,
        seq_len=500,
        in_channels=1,
        hidden_dim=256,
        latent_dim=32,
        num_classes=10
    ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(
                in_channels, 64,
                kernel_size=15,
                stride=1,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            _ResBlock1D(64, 64, stride=1),
            _ResBlock1D(64, 64, stride=1),
        )

        self.stage2 = nn.Sequential(
            _ResBlock1D(64, 128, stride=2),
            _ResBlock1D(128, 128, stride=1),
        )

        self.stage3 = nn.Sequential(
            _ResBlock1D(128, 256, stride=2),
            _ResBlock1D(256, 256, stride=1),
        )

        self.stage4 = nn.Sequential(
            _ResBlock1D(256, 512, stride=2),
            _ResBlock1D(512, 512, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc_hidden = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(
            hidden_dim, latent_dim
        )

        self.fc_logvar = nn.Linear(
            hidden_dim, latent_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        h = self.initial(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)

        h = self.gap(h).squeeze(-1)

        h = self.fc_hidden(h)

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
    "resnet1d": VIBResNet1D,
}


def build_vib(
    arch="mlp",
    input_shape=(1, 28, 28),
    hidden_dim=256,
    latent_dim=32,
    num_classes=10,
    seq_len=None,
    in_channels=None
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

    if arch == "resnet1d":

        return arch_cls(
            seq_len=seq_len,
            in_channels=in_channels,
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
