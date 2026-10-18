import numpy as np
from core.ray import Ray, RayState
from core.surfaces import Segment2D, SurfaceType
from core.intersections import trace_ray

def test_energy_balance():
    surfaces = [
        Segment2D([1, -1], [1, 1], reflectivity=0.5, surface_type=SurfaceType.WALL),  # Derecha
        Segment2D([-1, 1], [1, 1], reflectivity=1.0, surface_type=SurfaceType.RECEIVER), # Arriba
        Segment2D([-1, -1], [-1, 1], reflectivity=1.0, surface_type=SurfaceType.WALL), # Izquierda
        Segment2D([-1, -1], [1, -1], reflectivity=1.0, surface_type=SurfaceType.WALL)  # Abajo
    ]
    
    # Modificado: Se evita la diagonal perfecta que golpea la esquina [1,1].
    # Así aseguramos 100% que rebota primero a la derecha y luego sube al techo receptor.
    ray = Ray([0, 0], [1, 0.5], energy=100.0) 
    trace_ray(ray, surfaces, epsilon=1e-7)
    
    assert ray.state == RayState.CAPTURED
    assert ray.bounces == 1
    
    emitida = ray.initial_energy
    captada = ray.energy if ray.state == RayState.CAPTURED else 0.0
    absorbida = ray.initial_energy - ray.energy
    
    assert np.isclose(captada, 50.0)
    assert np.isclose(absorbida, 50.0)
    assert np.isclose(emitida, captada + absorbida)

def test_null_reflectivity_absorption():
    surfaces = [Segment2D([1, -1], [1, 1], reflectivity=0.0)]
    ray = Ray([0, 0], [1, 0], energy=100.0)
    trace_ray(ray, surfaces)
    
    assert ray.state == RayState.ABSORBED
    assert np.isclose(ray.energy, 0.0)
    assert ray.bounces == 1
def test_perfect_reflectivity_and_max_bounces():
    # Caja cerrada con 100% de reflectividad
    surfaces = [
        Segment2D([1, -1], [1, 1], reflectivity=1.0),
        Segment2D([-1, 1], [1, 1], reflectivity=1.0),
        Segment2D([-1, -1], [-1, 1], reflectivity=1.0),
        Segment2D([-1, -1], [1, -1], reflectivity=1.0)
    ]
    # Límite bajo de rebotes para probar interrupción numérica
    ray = Ray([0, 0], [1, 0.314], energy=100.0, max_bounces=15)
    trace_ray(ray, surfaces, epsilon=1e-7)
    
    # El rayo debe ser detenido por el límite, no por pérdida de energía
    assert ray.state == RayState.MAX_BOUNCES
    assert ray.bounces == 15
    # La energía debe permanecer intacta, demostrando conservación física pasiva
    assert np.isclose(ray.energy, 100.0)

def test_receiver_single_capture():
    # Un receptor en X=1
    surfaces = [Segment2D([1, -1], [1, 1], surface_type=SurfaceType.RECEIVER)]
    ray = Ray([0, 0], [1, 0], energy=100.0)
    trace_ray(ray, surfaces)
    
    # Comprobar que no atraviesa ni "rebota" dentro del receptor
    assert ray.state == RayState.CAPTURED
    assert ray.bounces == 0
    assert np.isclose(ray.energy, 100.0)
    
    # El historial debe tener exactamente 2 estados: Origen e Impacto (Capturado)
    assert len(ray.history) == 2