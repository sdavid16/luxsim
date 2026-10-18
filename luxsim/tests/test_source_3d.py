import numpy as np

from simulation.source import Source3D


def test_3d_source_spreads_rays_around_axis_instead_of_one_plane():
    axis = np.array([0.48, 0.24, 0.84], dtype=float)
    axis /= np.linalg.norm(axis)
    aperture_degrees = 100
    rays = Source3D([0, 0, 1], axis, aperture_degrees, 19, 1000).generate_rays()
    directions = np.array([ray.direction for ray in rays])

    assert len(rays) == 19
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-10)
    assert np.std(directions[:, 0]) > 0.05
    assert np.std(directions[:, 1]) > 0.05
    assert np.std(directions[:, 2]) > 0.01
    cos_angles = directions @ axis
    assert np.all(cos_angles >= np.cos(np.radians(aperture_degrees / 2)) - 1e-10)


def test_single_3d_ray_follows_source_axis():
    axis = np.array([1.0, 2.0, 3.0])
    rays = Source3D([0, 0, 0], axis, 120, 1, 100).generate_rays()
    np.testing.assert_allclose(rays[0].direction, axis / np.linalg.norm(axis))
