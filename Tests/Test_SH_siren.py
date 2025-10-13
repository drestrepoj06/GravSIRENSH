# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:25:51 2025

@author: jhonr
"""

import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SRC.Models.SH_siren import SH_SIREN

def test_sh_siren_forward_pass():
    # Instantiate model
    model = SH_SIREN(lmax=5, hidden_features=32, hidden_layers=2, out_features=1)

    # Example coordinates
    lon = torch.linspace(0, 360, 10)
    lat = torch.linspace(-90, 90, 10)

    # Forward pass
    output = model(lon, lat)

    # Assertions
    assert output is not None
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2
    assert output.shape[0] == 10
    assert output.shape[1] == 1