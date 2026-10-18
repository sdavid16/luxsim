import numpy as np
from core.surfaces import Ellipse2D
from visualization.plotter import interpolate_ray_history

def test_ellipse_intersection():
    # Elipse horizontal (Rx=4, Ry=2)
    ellipse = Ellipse2D([0, 0], 4.0, 2.0)
    
    # Rayo desde el centro hacia X positivo (debe chocar en t=4)
    t, n = ellipse.intersect([0, 0], [1, 0], 1e-7)
    assert np.isclose(t, 4.0)
    assert np.allclose(n, [-1, 0])  # La normal debe apuntar hacia adentro
    
    # Rayo hacia Y positivo (debe chocar en t=2)
    t, n = ellipse.intersect([0, 0], [0, 1], 1e-7)
    assert np.isclose(t, 2.0)
    assert np.allclose(n, [0, -1])

def test_animation_interpolator():
    # Simulamos un historial: Origen (0,0), Rebote (0,10) en dist=10, Fin (10,10) en dist=20
    history = [
        (np.array([0,0]), 100, "ACTIVO", 0.0),
        (np.array([0,10]), 100, "ACTIVO", 10.0),
        (np.array([10,10]), 100, "CAPTADO", 20.0)
    ]
    
    # Si la animación está en Tiempo=5 (mitad del primer segmento)
    pts = interpolate_ray_history(history, 5.0)
    assert len(pts) == 2
    assert np.allclose(pts[-1], [0, 5])
    
    # Si la animación está en Tiempo=15 (mitad del segundo segmento)
    pts2 = interpolate_ray_history(history, 15.0)
    assert len(pts2) == 3
    assert np.allclose(pts2[-1], [5, 10])