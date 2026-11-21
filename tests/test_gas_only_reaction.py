"""2 H <=> H2 with no condensed product is an ordinary reaction (T388 commit 1).

Before T388 the nucleation guard refused it ("Nucleation requires a condensed product"). The UV KKT
solver already handles a gas-only reaction; this checks it on T 1500-4000 K, P 1e4-1e8 Pa with
constant cv: no failed cell, mass action to 1e-5 in ln Kp at the returned T, H atoms conserved.
Kp: log10 Kp(bar) = 6.5003 - 24602.2/(T + 37) (antoine fit, 0.0017 dex over 1500-5000 K).
"""
import math

import pytest
import torch
from kintera import ThermoOptions, ThermoX, ThermoY, constants

A, B, C = 6.5003, 24602.2, 37.0
CARD = f"""reference-state: {{Tref: 0., Pref: 1.e5}}
species:
  - {{name: He, phase: gas, composition: {{He: 1}}, cv_R: 1.5}}
  - {{name: H2, phase: gas, composition: {{H: 2}}, cv_R: 2.5}}
  - {{name: H, phase: gas, composition: {{H: 1}}, cv_R: 1.5, u0_R: 28324.3}}
reactions:
  - {{equation: 2 H <=> H2, type: nucleation,
      rate-constant: {{formula: antoine, A: {A}, B: {B}, C: {C}}}}}
"""
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def log_kp(temp):  # ln Kp [Pa]
    return math.log(1e5) + (A - B / (temp + C)) * math.log(10.)


@pytest.mark.parametrize("device", DEVICES)
def test_h2_dissociation_equilibrium(tmp_path, device):
    (path := tmp_path / "h2diss.yaml").write_text(CARD)
    dev, R = torch.device(device), constants.Rgas
    th = ThermoY(ThermoOptions.from_yaml(str(path)))
    th.to(dev)
    inv_mu = th.buffer("inv_mu")
    T0, P0 = torch.meshgrid(torch.linspace(1500., 4000., 26, dtype=torch.float64, device=dev),
                            torch.logspace(4., 8., 41, dtype=torch.float64, device=dev), indexing="ij")
    T0, P0 = T0.reshape(-1), P0.reshape(-1)
    y = torch.zeros(2, T0.numel(), dtype=torch.float64, device=dev)
    y[0] = 0.75  # H2, H; He (species 0) is the rest
    rho = P0 / (R * T0 * ((1. - y.sum(0)) * inv_mu[0] + y[0] * inv_mu[1]))
    U = th.compute("VT->U", [th.compute("DY->V", [rho, y]), T0])
    diag, y1 = torch.zeros_like(rho).unsqueeze(-1), y.clone()
    th.forward(rho, U, y1, False, diag)
    V = th.compute("DY->V", [rho, y1])  # rho * y per species
    temp, c = th.compute("VU->T", [V, U]), V * inv_mu  # c [mol/m^3]
    res = 2 * torch.log(c[:, 2] * R * temp) - torch.log(c[:, 1] * R * temp) - log_kp(temp)
    assert int((diag < 0).sum()) == 0
    assert res.abs().max().item() <= 1e-5
    assert ((y1[0] + y1[1]) - (y[0] + y[1])).abs().max().item() <= 1e-14
    assert (c[:, 2] > 0).all()  # H appears

    part = ThermoY(ThermoOptions.from_yaml(str(path)).uv_solver("partition"))
    part.to(dev)
    with pytest.raises(RuntimeError, match="UV partition solver requires"):
        part.forward(rho, U, y.clone())
    tx = ThermoX(ThermoOptions.from_yaml(str(path)))  # the TP path takes it too
    tx.to(dev)
    x = torch.stack([torch.full_like(T0, .1), torch.full_like(T0, .9), torch.zeros_like(T0)], -1)
    tx.forward(T0, P0, x)
    assert (2 * torch.log(x[:, 2] * P0) - torch.log(x[:, 1] * P0) - log_kp(T0)).abs().max() <= 1e-5
