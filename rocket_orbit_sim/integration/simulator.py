"""Simulation orchestrator for multi-stage rocket ascent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from rocket_orbit_sim.physics.atmosphere import atmosphere_1976
from rocket_orbit_sim.physics.drag import CdTable, DEFAULT_CD_TABLE, drag_force
from rocket_orbit_sim.physics.guidance import PitchProgram, local_basis, thrust_direction
from rocket_orbit_sim.physics.orbit import MU_EARTH, orbital_elements

RE_EARTH = 6_378_137.0
G0 = 9.80665


@dataclass(frozen=True)
class Stage:
    name: str
    thrust: float
    isp: float
    prop_mass: float
    dry_mass: float
    diameter: float
    cd_table: CdTable

    @property
    def mdot(self) -> float:
        return 0.0 if self.thrust <= 0 else self.thrust / (self.isp * G0)

    @property
    def burn_time(self) -> float:
        return 0.0 if self.mdot == 0 else self.prop_mass / self.mdot

    @property
    def area_ref(self) -> float:
        return np.pi * (self.diameter / 2.0) ** 2


@dataclass(frozen=True)
class Mission:
    lat0: float
    lon0: float
    azimuth_deg: float
    pitch_program: PitchProgram
    t_max: float
    dt: float
    rtol: float
    atol: float
    stop_on_target: bool
    target_orbit: dict | None = None


@dataclass
class SegmentResult:
    t: np.ndarray
    y: np.ndarray
    stage_id: np.ndarray


def _dynamics(
    t: float,
    state: np.ndarray,
    stage: Stage,
    thrust_dir_fn: Callable[[float], np.ndarray],
    drag_enabled: bool,
) -> np.ndarray:
    r = state[0:3]
    v = state[3:6]
    mass = state[6]

    r_norm = np.linalg.norm(r)
    accel_grav = -MU_EARTH * r / (r_norm**3)

    thrust_accel = np.zeros(3)
    mass_flow = 0.0
    if stage.thrust > 0.0:
        thrust_dir = thrust_dir_fn(t)
        thrust_accel = stage.thrust * thrust_dir / mass
        mass_flow = stage.mdot

    drag_accel = np.zeros(3)
    if drag_enabled:
        drag_force_vec = drag_force(r, v, stage.area_ref, stage.cd_table)
        drag_accel = drag_force_vec / mass

    r_dot = v
    v_dot = accel_grav + thrust_accel + drag_accel
    m_dot = -mass_flow

    return np.concatenate([r_dot, v_dot, [m_dot]])


def _burnout_event(t0: float, burn_time: float) -> Callable:
    def event(t: float, _state: np.ndarray) -> float:
        return (t - t0) - burn_time

    event.terminal = True
    event.direction = 1
    return event


def _reentry_event(_t: float, state: np.ndarray) -> float:
    r = state[0:3]
    return np.linalg.norm(r) - RE_EARTH


def _integrate_segment(
    state0: np.ndarray,
    t0: float,
    t_end: float,
    stage: Stage,
    stage_id: int,
    thrust_dir_fn: Callable[[float], np.ndarray],
    dt: float,
    drag_enabled: bool,
    events: Iterable[Callable],
    rtol: float,
    atol: float,
) -> SegmentResult:
    t_eval = np.arange(t0, t_end + dt, dt)
    sol = solve_ivp(
        _dynamics,
        (t0, t_end),
        state0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        args=(stage, thrust_dir_fn, drag_enabled),
        events=list(events) if events else None,
    )

    stage_id_arr = np.full_like(sol.t, stage_id, dtype=int)
    return SegmentResult(t=sol.t, y=sol.y.T, stage_id=stage_id_arr)


def _stage_from_config(stage_cfg: dict) -> Stage:
    cd_table = stage_cfg.get("cd_table")
    if cd_table:
        cd_table = CdTable(
            mach=np.array([row[0] for row in cd_table], dtype=float),
            cd=np.array([row[1] for row in cd_table], dtype=float),
        )
    else:
        cd_table = DEFAULT_CD_TABLE

    return Stage(
        name=stage_cfg["name"],
        thrust=float(stage_cfg["thrust"]),
        isp=float(stage_cfg["isp"]),
        prop_mass=float(stage_cfg["prop_mass"]),
        dry_mass=float(stage_cfg["dry_mass"]),
        diameter=float(stage_cfg["diameter"]),
        cd_table=cd_table,
    )


def _assemble_dataframe(segments: list[SegmentResult]) -> pd.DataFrame:
    data = {
        "time": np.concatenate([seg.t for seg in segments]),
        "x": np.concatenate([seg.y[:, 0] for seg in segments]),
        "y": np.concatenate([seg.y[:, 1] for seg in segments]),
        "z": np.concatenate([seg.y[:, 2] for seg in segments]),
        "vx": np.concatenate([seg.y[:, 3] for seg in segments]),
        "vy": np.concatenate([seg.y[:, 4] for seg in segments]),
        "vz": np.concatenate([seg.y[:, 5] for seg in segments]),
        "m": np.concatenate([seg.y[:, 6] for seg in segments]),
        "stage_id": np.concatenate([seg.stage_id for seg in segments]),
    }

    df = pd.DataFrame(data)
    r = df[["x", "y", "z"]].to_numpy()
    v = df[["vx", "vy", "vz"]].to_numpy()
    speed = np.linalg.norm(v, axis=1)
    r_norm = np.linalg.norm(r, axis=1)
    h = r_norm - RE_EARTH

    rho = np.zeros_like(h)
    a = np.zeros_like(h)
    for idx, altitude in enumerate(h):
        props = atmosphere_1976(float(altitude))
        rho[idx] = props["rho"]
        a[idx] = props["a"]

    mach = speed / np.maximum(a, 1e-6)
    q = 0.5 * rho * speed**2

    df["h"] = h
    df["speed"] = speed
    df["Mach"] = mach
    df["q"] = q

    return df


def _target_orbit_thresholds(target_orbit: dict) -> tuple[float, float, float, float, float]:
    target_alt_m = float(target_orbit["altitude_km"]) * 1000.0
    target_a = RE_EARTH + target_alt_m
    target_incl = np.deg2rad(float(target_orbit["incl_deg"]))
    tolerance = target_orbit["tolerance"]
    tol_alt = float(tolerance["alt_km"]) * 1000.0
    tol_incl = np.deg2rad(float(tolerance["incl_deg"]))
    tol_ecc = float(tolerance["ecc"])
    return target_a, target_incl, tol_alt, tol_incl, tol_ecc


def _target_orbit_margin(r: np.ndarray, v: np.ndarray, target_orbit: dict) -> tuple[float, OrbitalElements]:
    elements = orbital_elements(r, v)
    target_a, target_incl, tol_alt, tol_incl, tol_ecc = _target_orbit_thresholds(target_orbit)
    margin = max(
        abs(elements.semi_major_axis - target_a) - tol_alt,
        elements.eccentricity - tol_ecc,
        abs(elements.inclination - target_incl) - tol_incl,
    )
    return margin, elements


def _target_orbit_event(target_orbit: dict) -> Callable:
    def event(_t: float, state: np.ndarray) -> float:
        r = state[0:3]
        v = state[3:6]
        margin, _elements = _target_orbit_margin(r, v, target_orbit)
        return margin

    event.terminal = True
    event.direction = -1
    return event


def run_simulation(vehicle_cfg: dict, mission_cfg: dict) -> pd.DataFrame:
    stage1_cfg = vehicle_cfg["stage1"]
    stage2_cfg = vehicle_cfg["stage2"]

    stage1 = _stage_from_config(stage1_cfg)
    stage2 = _stage_from_config(stage2_cfg)

    payload_mass = float(vehicle_cfg.get("payload_mass", stage2_cfg.get("payload_mass", 0.0)))
    stage2_total = stage2.prop_mass + stage2.dry_mass + payload_mass
    stage1_total = stage1.prop_mass + stage1.dry_mass + stage2_total

    lat0 = np.deg2rad(mission_cfg["launch_site"]["lat_deg"])
    lon0 = np.deg2rad(mission_cfg["launch_site"]["lon_deg"])

    guidance_cfg = mission_cfg["guidance"]
    t_vertical = guidance_cfg.get("t_vertical_s", guidance_cfg.get("t_vertical"))
    if t_vertical is None:
        raise KeyError("guidance.t_vertical_s or guidance.t_vertical is required")

    if "pitch_schedule_deg" in guidance_cfg:
        points = np.array(guidance_cfg["pitch_schedule_deg"], dtype=float)
        points[:, 1] = 90.0 - points[:, 1]
    else:
        points = np.array(guidance_cfg["pitch_points"], dtype=float)

    pitch_program = PitchProgram(
        t_vertical=float(t_vertical),
        points=points,
    )

    sim_cfg = mission_cfg.get("sim", mission_cfg.get("simulation"))
    if sim_cfg is None:
        raise KeyError("sim or simulation configuration is required")

    t_max = sim_cfg.get("t_max_s", sim_cfg.get("t_max"))
    if t_max is None:
        raise KeyError("sim.t_max_s or sim.t_max is required")

    mission = Mission(
        lat0=lat0,
        lon0=lon0,
        azimuth_deg=float(guidance_cfg["azimuth_deg"]),
        pitch_program=pitch_program,
        t_max=float(t_max),
        dt=float(sim_cfg.get("dt", 1.0)),
        rtol=float(sim_cfg.get("rtol", 1e-7)),
        atol=float(sim_cfg.get("atol", 1e-9)),
        stop_on_target=bool(sim_cfg.get("stop_on_target", False)),
        target_orbit=mission_cfg.get("target_orbit"),
    )

    basis = local_basis(mission.lat0, mission.lon0)

    def thrust_dir_fn(t: float) -> np.ndarray:
        pitch = mission.pitch_program.pitch(t)
        return thrust_direction(pitch, mission.azimuth_deg, basis)

    r0 = RE_EARTH * np.array([
        np.cos(lat0) * np.cos(lon0),
        np.cos(lat0) * np.sin(lon0),
        np.sin(lat0),
    ])
    v0 = np.zeros(3)
    state = np.concatenate([r0, v0, [stage1_total]])

    segments: list[SegmentResult] = []
    t0 = 0.0

    stage1_events = [_burnout_event(t0, stage1.burn_time), _reentry_event]
    if mission.stop_on_target and mission.target_orbit:
        stage1_events.append(_target_orbit_event(mission.target_orbit))
    segments.append(
        _integrate_segment(
            state,
            t0,
            mission.t_max,
            stage1,
            1,
            thrust_dir_fn,
            mission.dt,
            drag_enabled=True,
            events=stage1_events,
            rtol=mission.rtol,
            atol=mission.atol,
        )
    )

    state = segments[-1].y[-1].copy()
    t0 = segments[-1].t[-1]
    h_end = np.linalg.norm(state[0:3]) - RE_EARTH
    if h_end <= 0 or t0 >= mission.t_max:
        return _assemble_dataframe(segments)

    if mission.stop_on_target and mission.target_orbit:
        margin, elements = _target_orbit_margin(state[0:3], state[3:6], mission.target_orbit)
        if margin <= 0:
            df = _assemble_dataframe(segments)
            df.attrs["orbit_elements"] = elements
            df.attrs["target_orbit_achieved"] = True
            return df

    state[6] -= stage1.dry_mass

    stage2_events = [_burnout_event(t0, stage2.burn_time), _reentry_event]
    if mission.stop_on_target and mission.target_orbit:
        stage2_events.append(_target_orbit_event(mission.target_orbit))
    segments.append(
        _integrate_segment(
            state,
            t0,
            mission.t_max,
            stage2,
            2,
            thrust_dir_fn,
            mission.dt,
            drag_enabled=True,
            events=stage2_events,
            rtol=mission.rtol,
            atol=mission.atol,
        )
    )

    state = segments[-1].y[-1].copy()
    t0 = segments[-1].t[-1]
    h_end = np.linalg.norm(state[0:3]) - RE_EARTH
    if h_end <= 0 or t0 >= mission.t_max:
        return _assemble_dataframe(segments)

    if mission.stop_on_target and mission.target_orbit:
        margin, elements = _target_orbit_margin(state[0:3], state[3:6], mission.target_orbit)
        if margin <= 0:
            df = _assemble_dataframe(segments)
            df.attrs["orbit_elements"] = elements
            df.attrs["target_orbit_achieved"] = True
            return df

    coast_stage = Stage(
        name="coast",
        thrust=0.0,
        isp=1.0,
        prop_mass=0.0,
        dry_mass=0.0,
        diameter=stage2.diameter,
        cd_table=stage2.cd_table,
    )

    coast_events = [_reentry_event]
    if mission.stop_on_target and mission.target_orbit:
        coast_events.append(_target_orbit_event(mission.target_orbit))
    segments.append(
        _integrate_segment(
            state,
            t0,
            mission.t_max,
            coast_stage,
            3,
            thrust_dir_fn,
            mission.dt,
            drag_enabled=True,
            events=coast_events,
            rtol=mission.rtol,
            atol=mission.atol,
        )
    )

    df = _assemble_dataframe(segments)

    if mission.target_orbit:
        r = df[["x", "y", "z"]].to_numpy()
        v = df[["vx", "vy", "vz"]].to_numpy()
        if mission.stop_on_target:
            margin, elements = _target_orbit_margin(r[-1], v[-1], mission.target_orbit)
            df.attrs["target_orbit_achieved"] = margin <= 0
        else:
            elements = orbital_elements(r[-1], v[-1])
        df.attrs["orbit_elements"] = elements

    return df
