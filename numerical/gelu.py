"""Implementation of Gelu network used in Martin & Schaub (2022)
jhonr"""
from torch import nn

# Based on simplified implementation of https://github.com/esa/masconCube/blob/main/mascon_cube/pinn_gm/_network.py
# And https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html

class Gelu(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
        ]

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