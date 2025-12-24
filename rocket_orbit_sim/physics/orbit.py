"""Orbital mechanics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU_EARTH = 3.986004418e14


@dataclass(frozen=True)
class OrbitalElements:
    semi_major_axis: float
    eccentricity: float
    inclination: float
    r_perigee: float
    r_apogee: float


def orbital_elements(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> OrbitalElements:
    """Compute classical orbital elements from position and velocity."""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)

    if h_norm == 0.0:
        return OrbitalElements(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    e_vec = (np.cross(v, h_vec) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    energy = v_norm * v_norm / 2 - mu / r_norm
    a = -mu / (2 * energy) if energy != 0.0 else float("inf")

    inclination = np.arccos(h_vec[2] / h_norm)
    r_perigee = a * (1 - e)
    r_apogee = a * (1 + e)

    return OrbitalElements(a, e, inclination, r_perigee, r_apogee)
