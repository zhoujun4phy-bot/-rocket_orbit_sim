"""US Standard Atmosphere 1976 (0-86 km) approximation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

G0 = 9.80665
R_AIR = 287.05287
GAMMA_AIR = 1.4


@dataclass(frozen=True)
class AtmosLayer:
    h_base: float
    t_base: float
    p_base: float
    lapse: float


def _build_layers() -> list[AtmosLayer]:
    # Layer boundaries (m), lapse rates (K/m), base temperature (K) at sea level
    boundaries = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 86000.0]
    lapse_rates = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
    t0 = 288.15
    p0 = 101325.0

    layers: list[AtmosLayer] = []
    t_base = t0
    p_base = p0

    for idx, lapse in enumerate(lapse_rates):
        h_base = boundaries[idx]
        h_top = boundaries[idx + 1]
        layers.append(AtmosLayer(h_base=h_base, t_base=t_base, p_base=p_base, lapse=lapse))
        if lapse == 0.0:
            p_base = p_base * exp(-G0 * (h_top - h_base) / (R_AIR * t_base))
        else:
            t_top = t_base + lapse * (h_top - h_base)
            p_base = p_base * (t_top / t_base) ** (-G0 / (lapse * R_AIR))
            t_base = t_top
    return layers


_LAYERS = _build_layers()


def atmosphere_1976(h: float) -> dict[str, float]:
    """Return atmosphere properties at geometric altitude h (m)."""
    if h < 0.0:
        h = 0.0
    if h > 86000.0:
        h = 86000.0

    layer = _LAYERS[-1]
    for candidate in _LAYERS:
        if h >= candidate.h_base:
            layer = candidate
        else:
            break

    h_delta = h - layer.h_base
    if layer.lapse == 0.0:
        t = layer.t_base
        p = layer.p_base * exp(-G0 * h_delta / (R_AIR * t))
    else:
        t = layer.t_base + layer.lapse * h_delta
        p = layer.p_base * (t / layer.t_base) ** (-G0 / (layer.lapse * R_AIR))

    rho = p / (R_AIR * t)
    a = sqrt(GAMMA_AIR * R_AIR * t)
    return {"T": t, "p": p, "rho": rho, "a": a}


def sample_profile(altitudes: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized sampling of atmosphere properties for altitudes array."""
    temps = np.zeros_like(altitudes, dtype=float)
    press = np.zeros_like(altitudes, dtype=float)
    dens = np.zeros_like(altitudes, dtype=float)
    sound = np.zeros_like(altitudes, dtype=float)

    for idx, h in np.ndenumerate(altitudes):
        props = atmosphere_1976(float(h))
        temps[idx] = props["T"]
        press[idx] = props["p"]
        dens[idx] = props["rho"]
        sound[idx] = props["a"]

    return {"T": temps, "p": press, "rho": dens, "a": sound}
