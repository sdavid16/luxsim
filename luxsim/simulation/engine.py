"""Orquestación del trazado geométrico de los rayos."""

from core.intersections import trace_ray


def run_simulation(source, surfaces, max_bounces=50, epsilon=1e-7, max_distance=40.0):
    """Genera los rayos y devuelve sus historiales de trayectoria.

    Args:
        source: fuente 2D o 3D con un método generate_rays.
        surfaces: superficies analíticas que pueden intersectar los rayos.
        max_bounces: límite de reflexiones por rayo.
        epsilon: tolerancia numérica para intersecciones cercanas.
        max_distance: longitud total máxima recorrida por cada rayo.
    """
    rays = source.generate_rays()

    for ray in rays:
        ray.max_bounces = max_bounces
        trace_ray(ray, surfaces, epsilon, max_distance=max_distance)

    return rays
