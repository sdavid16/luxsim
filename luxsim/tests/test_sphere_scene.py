import numpy as np
import plotly.graph_objects as go

from environments.scenes import EnvironmentLibrary
from visualization.plotter import plot_scene


def test_sphere_scene_uses_equal_ellipsoid_radii_and_renders_in_3d():
    surfaces = EnvironmentLibrary.get_env_C_ellipsoid(5, 5, 5, 1.0, 2)
    sphere = next(surface for surface in surfaces if type(surface).__name__ == "Ellipsoid3D")

    assert np.isclose(sphere.rx, sphere.ry)
    assert np.isclose(sphere.ry, sphere.rz)

    fig = plot_scene(surfaces, [], is_3d=True)
    sphere_trace = next(
        trace for trace in fig.data
        if isinstance(trace, go.Surface) and trace.name == "Esfera"
    )
    assert sphere_trace.x.shape == sphere_trace.y.shape == sphere_trace.z.shape
    assert fig.layout.scene.aspectmode == "data"
