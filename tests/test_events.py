import numpy as np

from rocket_orbit_sim.integration.simulator import run_simulation


def test_stage_events_mass_and_id():
    vehicle = {
        "stage1": {
            "name": "S1",
            "thrust": 2.0e5,
            "isp": 200.0,
            "prop_mass": 2.0e4,
            "dry_mass": 2.0e3,
            "diameter": 2.5,
        },
        "stage2": {
            "name": "S2",
            "thrust": 5.0e4,
            "isp": 300.0,
            "prop_mass": 5.0e3,
            "dry_mass": 1.0e3,
            "diameter": 2.0,
        },
        "payload_mass": 500.0,
    }

    mission = {
        "launch_site": {"lat_deg": 0.0, "lon_deg": 0.0},
        "guidance": {
            "azimuth_deg": 90.0,
            "t_vertical": 5.0,
            "pitch_points": [[5.0, 0.0], [30.0, 10.0]],
        },
        "simulation": {"t_max": 200.0, "dt": 1.0},
    }

    df = run_simulation(vehicle, mission)

    assert set(df["stage_id"].unique()) == {1, 2, 3}

    stage1_last = df[df["stage_id"] == 1].iloc[-1]
    stage2_first = df[df["stage_id"] == 2].iloc[0]

    expected_stage1_end_mass = (
        vehicle["stage1"]["dry_mass"]
        + vehicle["stage2"]["prop_mass"]
        + vehicle["stage2"]["dry_mass"]
        + vehicle["payload_mass"]
    )
    assert np.isclose(stage1_last["m"], expected_stage1_end_mass, rtol=1e-3)

    expected_stage2_start_mass = (
        vehicle["stage2"]["prop_mass"]
        + vehicle["stage2"]["dry_mass"]
        + vehicle["payload_mass"]
    )
    assert np.isclose(stage2_first["m"], expected_stage2_start_mass, rtol=1e-3)
