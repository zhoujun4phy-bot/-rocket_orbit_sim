"""Command line entry point for rocket orbit simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rocket_orbit_sim.config import load_yaml
from rocket_orbit_sim.integration.simulator import run_simulation
from rocket_orbit_sim.visualization.plots import plot_groundtrack_mercator, plot_trajectory_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage rocket orbit simulation")
    parser.add_argument("--mission", required=True, help="Mission YAML config")
    parser.add_argument("--vehicle", required=True, help="Vehicle YAML config")
    parser.add_argument("--output", default="results", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mission_cfg = load_yaml(args.mission)
    vehicle_cfg = load_yaml(args.vehicle)

    df = run_simulation(vehicle_cfg, mission_cfg)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "trajectory.csv"
    df.to_csv(csv_path, index=False)

    r = df[["x", "y", "z"]].to_numpy()
    plot_trajectory_3d(r, str(output_dir / "trajectory_3d.png"))
    plot_groundtrack_mercator(r, str(output_dir / "groundtrack_mercator.png"))

    print(f"Saved trajectory to {csv_path}")


if __name__ == "__main__":
    main()
