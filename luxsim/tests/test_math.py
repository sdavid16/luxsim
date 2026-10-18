import numpy as np
import pytest
from core.vectors import normalize, norm, reflect
from core.ray import Ray
from core.surfaces import Segment2D, Circle2D, RectangularPlane3D
from core.intersections import calculate_closest_intersection

def test_vector_normalization():
    v = np.array([3, 4, 0])
    v_norm = normalize(v)
    assert np.isclose(norm(v_norm), 1.0)
    assert np.allclose(v_norm, [0.6, 0.8, 0])

def test_segment_intersection_and_tangency():
    seg = Segment2D([1, -1], [1, 1])
    
    # Impacto normal perpendicular
    t, n = seg.intersect([0, 0], [1, 0], 1e-7)
    assert np.isclose(t, 1.0)
    assert np.allclose(n, [-1, 0])
    
    # Rayo paralelo (no cruza)
    assert seg.intersect([0, 0], [0, 1], 1e-7) is None
    
    # Rayo pasando justo por una esquina / vértice (Tangencia analítica)
    t, n = seg.intersect([0, -1], [1, 0], 1e-7)
    assert np.isclose(t, 1.0)

def test_circle_from_surface_no_self_intersect():
    circle = Circle2D([0, 0], 1.0)
    # Rayo nace exactamente sobre el borde del círculo apuntando hacia afuera
    # El epsilon (1e-7) debe impedir que t=0 se detecte como intersección
    t_n = circle.intersect([1, 0], [1, 0], epsilon=1e-7)
    assert t_n is None

def test_plane_3d_intersection():
    # Plano en Z=5, tamaño 10x10
    plane = RectangularPlane3D([0,0,5], [1,0,0], [0,1,0], 10, 10)
    
    # Rayo choca en el centro
    t, n = plane.intersect([0,0,0], [0,0,1], 1e-7)
    assert np.isclose(t, 5.0)
    assert np.allclose(n, [0,0,-1])
    
    # Rayo pasa por fuera del rectángulo de 10x10
    assert plane.intersect([0, 6, 0], [0,0,1], 1e-7) is None
from core.surfaces import CylinderWall3D
from core.intersections import calculate_closest_intersection

def test_circle_intersections():
    circle = Circle2D([0, 0], 2.0)
    
    # 1. Secante (Rayo atraviesa el círculo, debe devolver el impacto más cercano)
    t, n = circle.intersect([-5, 0], [1, 0], 1e-7)
    assert np.isclose(t, 3.0)  # Impacto en X=-2
    assert np.allclose(n, [-1, 0])
    
    # 2. Tangencia perfecta (b^2 - 4ac = 0)
    t, n = circle.intersect([-5, 2], [1, 0], 1e-7)
    assert np.isclose(t, 5.0)  # Impacto en [0, 2]
    assert np.allclose(n, [0, 1])
    
    # 3. Ausencia de impacto (Pasa por encima)
    assert circle.intersect([-5, 3], [1, 0], 1e-7) is None

def test_closest_impact_selection():
    # Tres paredes paralelas en X=1, X=2, X=3
    surfaces = [
        Segment2D([3, -1], [3, 1]),
        Segment2D([1, -1], [1, 1]),
        Segment2D([2, -1], [2, 1])
    ]
    ray = Ray([0, 0], [1, 0])
    t, surface, normal = calculate_closest_intersection(ray, surfaces, 1e-7)
    
    # Debe seleccionar X=1, aunque la lista tenga X=3 primero
    assert np.isclose(t, 1.0)
    assert surface == surfaces[1]

def test_ray_looking_away_and_origin_on_surface():
    plane = RectangularPlane3D([0, 0, 5], [1, 0, 0], [0, 1, 0], 10, 10)
    
    # Rayo nace frente al plano pero apunta en dirección contraria (-Z)
    assert plane.intersect([0, 0, 0], [0, 0, -1], 1e-7) is None
    
    # Rayo nace exactamente SOBRE el plano apuntando hacia afuera (+Z)
    # El epsilon debe evitar que t=0 se detecte como intersección
    assert plane.intersect([0, 0, 5], [0, 0, 1], 1e-7) is None

def test_cylinder_finite_bounds():
    cyl = CylinderWall3D([0, 0, 0], 2.0, 5.0)  # Cilindro de altura 5
    
    # Rayo choca con la pared dentro de los límites Z (Z=2)
    t, n = cyl.intersect([-5, 0, 2], [1, 0, 0], 1e-7)
    assert np.isclose(t, 3.0)
    
    # Rayo apunta al cilindro pero pasa por encima (Z=6)
    assert cyl.intersect([-5, 0, 6], [1, 0, 0], 1e-7) is None

def test_reflection_law_angles():
    # Verificar explícitamente que el ángulo de incidencia = ángulo de reflexión
    v_in = normalize([1, -1])  # Rayo incide a 45 grados hacia abajo-derecha
    normal = [0, 1]            # Pared horizontal (suelo), normal hacia arriba
    v_out = reflect(v_in, normal)
    
    # Debe salir a 45 grados hacia arriba-derecha
    assert np.allclose(v_out, normalize([1, 1]))