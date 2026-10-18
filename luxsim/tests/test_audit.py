import numpy as np
import pytest
from core.surfaces import Segment2D
from simulation.source import Source2D
from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation
from simulation.analysis import calculate_heatmap_2d, grid_search_optimization
from visualization.plotter import interpolate_ray_history
from core.ray import RayState

def test_audit_environments():
    # Comprueba que todos los ambientes se instancian correctamente con sus geometrías
    assert len(EnvironmentLibrary.get_env_A_circular(5, 0.8, 3)) == 2
    assert len(EnvironmentLibrary.get_env_A_cylindrical(5, 10, 0.8, 8)) == 4
    assert len(EnvironmentLibrary.get_env_B_rectangular(10, 10, 0.8, 4)) == 5
    assert len(EnvironmentLibrary.get_env_C_elliptical(6, 4, 0.8, 2)) == 2
    assert len(EnvironmentLibrary.get_env_C_ellipsoid(6, 4, 8, 0.8, 4)) == 2
    assert len(EnvironmentLibrary.get_env_D_gallery(10, 10, 0.8)) == 5
    assert len(EnvironmentLibrary.get_env_E_obstacle(10, 10, 0.8)) == 6
    assert len(EnvironmentLibrary.get_env_F_guide(20, 5, 0.8)) == 3
    assert len(EnvironmentLibrary.get_env_G_mixed_ref(10, 10)) == 5
    assert len(EnvironmentLibrary.get_env_H_multi_receiver(5, 0.8)) == 3

def test_audit_heatmap():
    # Verifica que el mapa de calor procesa rayos reales
    surfaces = EnvironmentLibrary.get_env_B_rectangular(10, 10, 1.0, 4)
    source = Source2D([0, -2], 90, 60, 10, 100.0)
    rays, metrics = run_simulation(source, surfaces)
    
    # Calor en 10 celdas
    hist, cv = calculate_heatmap_2d(rays, bins=10)
    
    assert len(hist) == 10
    # La energía total mapeada debe coincidir estricta y termodinámicamente con la captada
    assert np.isclose(np.sum(hist), metrics['e_captada'])

def test_audit_optimization():
    surfaces_func = lambda: EnvironmentLibrary.get_env_B_rectangular(10, 10, 1.0, 4)
    
    # Evaluamos 3 configuraciones de emisión: Derecha(0), Arriba(90) e Izquierda(180)
    source_func = lambda ang: Source2D([0, -2], ang, 10, 5, 100.0)
    
    # El optimizador DEBE encontrar que disparar a 90 grados (hacia el receptor en Y=4) es lo mejor
    best_ang, best_val = grid_search_optimization(source_func, surfaces_func, 'e_captada', [0, 90, 180])
    
    assert best_ang == 90
    assert best_val > 0

def test_audit_animation_independence():
    # Verificar que el truncamiento de la animación no daña las variables físicas
    source = Source2D([0, -2], 90, 60, 1, 100.0)
    surfaces = EnvironmentLibrary.get_env_B_rectangular(10, 10, 1.0, 4)
    rays, metrics = run_simulation(source, surfaces)
    
    ray = rays[0]
    original_energy = ray.energy
    original_history_len = len(ray.history)
    
    # Simular que el usuario pone la barra de animación temporal a la mitad
    max_d = ray.cumulative_distance / 2.0
    pts = interpolate_ray_history(ray.history, max_d)
    
    # El rayo debe seguir intacto termodinámicamente (Independencia)
    assert ray.energy == original_energy
    assert len(ray.history) == original_history_len
    # Pero los puntos visualizados deben ser estrictamente menores o estar truncados
    assert len(pts) < original_history_len or (len(pts) == original_history_len and np.linalg.norm(pts[-1] - ray.history[-1][0]) > 0.01)