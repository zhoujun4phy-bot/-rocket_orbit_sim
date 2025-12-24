import numpy as np

from rocket_orbit_sim.physics.atmosphere import atmosphere_1976


def test_density_decreases_with_altitude():
    altitudes = np.linspace(0, 80_000, 9)
    densities = [atmosphere_1976(h)["rho"] for h in altitudes]
    assert all(densities[i] > densities[i + 1] for i in range(len(densities) - 1))


def test_layer_continuity():
    boundaries = [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]
    for h in boundaries[1:]:
        rho_lower = atmosphere_1976(h - 1.0)["rho"]
        rho_upper = atmosphere_1976(h + 1.0)["rho"]
        assert np.isclose(rho_lower, rho_upper, rtol=2e-2)
