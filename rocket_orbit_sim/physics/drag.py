"""Aerodynamic drag models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import atmosphere_1976


@dataclass(frozen=True)
class CdTable:
    mach: np.ndarray
    cd: np.ndarray

    def evaluate(self, mach: float) -> float:
        return float(np.interp(mach, self.mach, self.cd))


DEFAULT_CD_TABLE = CdTable(
    mach=np.array([0.0, 0.5, 0.9, 1.0, 1.2, 2.0, 5.0, 8.0, 12.0]),
    cd=np.array([0.3, 0.32, 0.35, 0.5, 0.4, 0.3, 0.25, 0.23, 0.22]),
)


def drag_force(r: np.ndarray, v: np.ndarray, area_ref: float, cd_table: CdTable) -> np.ndarray:
    """Compute drag force vector for position r and velocity v."""
    speed = np.linalg.norm(v)
    if speed == 0.0:
        return np.zeros(3)

    h = np.linalg.norm(r) - 6_378_137.0
    props = atmosphere_1976(h)
    rho = props["rho"]
    mach = speed / props["a"] if props["a"] > 0.0 else 0.0
    cd = cd_table.evaluate(mach)

    drag_mag = 0.5 * rho * cd * area_ref * speed * speed
    return -drag_mag * (v / speed)
