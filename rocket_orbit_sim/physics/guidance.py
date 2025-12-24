"""Simple pitch program guidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchProgram:
    t_vertical: float
    points: np.ndarray  # columns: time, pitch_deg

    def pitch(self, t: float) -> float:
        if t < self.t_vertical:
            return 0.0
        times = self.points[:, 0]
        pitches = self.points[:, 1]
        return float(np.interp(t, times, pitches, left=pitches[0], right=pitches[-1]))


def local_basis(lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (er, enorth, eeast) unit vectors at given lat/lon."""
    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)

    er = np.array([clat * clon, clat * slon, slat])
    enorth = np.array([-slat * clon, -slat * slon, clat])
    eeast = np.array([-slon, clon, 0.0])
    return er, enorth, eeast


def thrust_direction(pitch_deg: float, azimuth_deg: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Compute thrust direction in ECI from pitch and azimuth."""
    pitch = np.deg2rad(pitch_deg)
    azimuth = np.deg2rad(azimuth_deg)
    er, enorth, eeast = basis

    lateral = np.cos(azimuth) * enorth + np.sin(azimuth) * eeast
    direction = np.cos(pitch) * er + np.sin(pitch) * lateral
    return direction / np.linalg.norm(direction)
