# Rocket Orbit Simulation (Two-Stage)

A research-readable, extensible, and reproducible two-stage rocket ascent simulation that delivers a satellite toward LEO in a simplified ECI frame (no Earth rotation).

## Features
- Modular layers: **physics models**, **event-driven integration**, **visualization**.
- US Standard Atmosphere 1976 (0–86 km) for `rho(h), T(h), p(h), a(h)`.
- Two-stage vehicle with burnout and separation events.
- Simple pitch program guidance with fixed launch azimuth.
- Outputs: trajectory CSV + 3D trajectory plot + Mercator ground track.

## Assumptions
- **Inertial frame**: ECI approximation without Earth rotation.
- Vehicle is a **point mass**; two-stage with instantaneous separation.
- Drag on a **cylindrical body** using reference area `A = π(D/2)^2`.
- Drag coefficient model is a **Mach-based table** (easy to replace).

## Units
- Length: meters (m)
- Time: seconds (s)
- Mass: kilograms (kg)
- Angles: degrees in configs, internally radians

## Repository Structure
```
rocket_orbit_sim/
  physics/          # Atmosphere, drag, guidance, orbit helpers
  integration/      # Stage definitions, events, solve_ivp orchestration
  visualization/    # Matplotlib plotting utilities
config/             # Example vehicle + mission configurations
results/            # Simulation outputs
```

## Quickstart
Install dependencies:
```bash
pip install numpy scipy pyyaml pandas matplotlib
```

Run a simulation:
```bash
python -m rocket_orbit_sim.simulate \
  --mission config/mission.yaml \
  --vehicle config/vehicle.yaml
```

Outputs will be written to `results/`:
- `trajectory.csv`
- `trajectory_3d.png`
- `groundtrack_mercator.png`

## Configuration
### Vehicle (`config/vehicle.yaml`)
```yaml
stage1:
  thrust: 5.0e6
  isp: 260
  prop_mass: 2.5e5
  dry_mass: 3.0e4
  diameter: 4.0
  cd_table:
    - [0.0, 0.3]
    - [1.0, 0.5]
    - [5.0, 0.25]
stage2:
  thrust: 1.0e6
  isp: 320
  prop_mass: 6.0e4
  dry_mass: 8.0e3
  diameter: 3.0
payload_mass: 3.0e3
```

### Mission (`config/mission.yaml`)
```yaml
launch_site:
  lat_deg: 28.5
  lon_deg: -80.6

guidance:
  azimuth_deg: 90.0
  t_vertical: 10.0
  pitch_points:
    - [10.0, 0.0]
    - [60.0, 20.0]
    - [120.0, 45.0]

simulation:
  t_max: 800.0
  dt: 1.0
```

## Extending the Model
- **Aerodynamics**: replace `CdTable` in `rocket_orbit_sim/physics/drag.py` with a higher-fidelity `Cd(M, Re)` model or table.
- **Guidance**: customize `PitchProgram` or implement a closed-loop controller in `rocket_orbit_sim/physics/guidance.py`.
- **Events**: add new termination logic in `rocket_orbit_sim/integration/simulator.py`.

## Testing
Run tests with:
```bash
pytest
```

## Outputs
The simulation writes per-time-step data including position, velocity, mass, altitude, speed, Mach, dynamic pressure, and stage ID.

## License
MIT (add your preferred license if needed).
