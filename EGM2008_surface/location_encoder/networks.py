"""Implementation of SirenNET and LinearNet
jhonr"""

# From https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
from torch import nn
import torch
import math

# See paper Sitzmann et al., (2020), sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

# If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
# nonlinearity. Different signals may require different omega_0 in the first layer - this is a
# hyperparameter.
    
# If is_first=False, then the weights will be divided by omega_0 to keep the magnitude of
# activations constant, but boost gradients to the weight matrix.

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIRENNet(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 first_omega_0=30, hidden_omega_0=1.0, final_linear=True):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        if final_linear:
            layers.append(nn.Linear(hidden_features, out_features))
        else:
            layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LINEARNet(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_features, hidden_features))

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))

        layers.append(nn.Linear(hidden_features, out_features))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class GELUNet(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.GELU())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_features, out_features))

        self.model = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)