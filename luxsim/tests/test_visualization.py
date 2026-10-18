import numpy as np
import plotly.graph_objects as go

from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation
from simulation.source import Source2D, Source3D
from visualization.plotter import interpolate_ray_history, plot_scene


def test_2d_scene_renders_segments_and_curves():
    surfaces = EnvironmentLibrary.get_env_A_circular(5, 0.8, 3)
    fig = plot_scene(surfaces, [], is_3d=False)

    assert len(fig.data) >= 2
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.height >= 500


def test_3d_cylindrical_scene_renders_actual_surfaces():
    surfaces = EnvironmentLibrary.get_env_A_cylindrical(5, 10, 0.8, 8)
    source = Source3D([0, 0, 1], [0, 0, 1], 30, 12, 1000)
    rays, _ = run_simulation(source, surfaces, max_bounces=8)
    fig = plot_scene(surfaces, rays, is_3d=True)

    assert any(isinstance(trace, go.Surface) for trace in fig.data)
    assert any(isinstance(trace, go.Scatter3d) for trace in fig.data)
    assert fig.layout.scene.zaxis.title.text == "Z"
    assert fig.layout.scene.aspectmode == "data"


def test_3d_ellipsoid_scene_renders_actual_surfaces():
    surfaces = EnvironmentLibrary.get_env_C_ellipsoid(6, 4, 8, 0.8, 2)
    source = Source3D([0, 0, 0], [0, 0, 1], 20, 8, 1000)
    rays, _ = run_simulation(source, surfaces, max_bounces=8)
    fig = plot_scene(surfaces, rays, is_3d=True)

    assert any(isinstance(trace, go.Surface) for trace in fig.data)
    assert any(isinstance(trace, go.Scatter3d) for trace in fig.data)


def test_ray_history_interpolation_stops_at_requested_distance():
    source = Source2D([0, 0], 0, 0, 1, 100)
    rays = source.generate_rays()
    ray = rays[0]
    ray.record_state(np.array([10.0, 0.0]))

    pts = interpolate_ray_history(ray.history, 4.0)

    assert pts.shape == (2, 2)
    np.testing.assert_allclose(pts[-1], [4.0, 0.0])


def test_every_declared_2d_environment_can_be_plotted():
    environments = [
        EnvironmentLibrary.get_env_A_circular(5, 0.8, 3),
        EnvironmentLibrary.get_env_B_rectangular(10, 10, 0.8, 4),
        EnvironmentLibrary.get_env_C_elliptical(6, 4, 0.8, 2),
        EnvironmentLibrary.get_env_D_gallery(10, 10, 0.8),
        EnvironmentLibrary.get_env_E_obstacle(10, 10, 0.8),
        EnvironmentLibrary.get_env_F_guide(10, 4, 0.8),
        EnvironmentLibrary.get_env_G_mixed_ref(10, 10),
        EnvironmentLibrary.get_env_H_multi_receiver(5, 0.8),
    ]

    for surfaces in environments:
        fig = plot_scene(surfaces, [], is_3d=False)
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "X"
