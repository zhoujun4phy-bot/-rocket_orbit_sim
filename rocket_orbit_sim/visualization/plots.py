"""Visualization helpers for trajectory results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

RE_EARTH = 6_378_137.0


def plot_trajectory_3d(r: np.ndarray, output_path: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(r[:, 0], r[:, 1], r[:, 2], label="Trajectory", color="tab:orange")

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = RE_EARTH * np.outer(np.cos(u), np.sin(v))
    y = RE_EARTH * np.outer(np.sin(u), np.sin(v))
    z = RE_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=0.4, linewidth=0)

    ax.set_xlabel("ECI X (m)")
    ax.set_ylabel("ECI Y (m)")
    ax.set_zlabel("ECI Z (m)")
    ax.set_title("3D Trajectory")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_groundtrack_mercator(r: np.ndarray, output_path: str) -> None:
    r_norm = np.linalg.norm(r, axis=1)
    lon = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    lat = np.arcsin(r[:, 2] / r_norm)

    x = RE_EARTH * lon
    y = RE_EARTH * np.log(np.tan(np.pi / 4 + lat / 2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="tab:green")
    ax.set_xlabel("Mercator X (m)")
    ax.set_ylabel("Mercator Y (m)")
    ax.set_title("Ground Track (Mercator)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
