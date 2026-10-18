import numpy as np
import pytest
from simulation.source import Source2D, Source3D
from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation

def test_integration_2d_energy_balance():
    # Fuente direccional disparando contra un recinto con receptor y reflectividad parcial
    source = Source2D(position=[0, -2], angle_center=90, angle_spread=45, num_rays=10, total_energy=100.0)
    
    # Usando la nueva sintaxis de EnvironmentLibrary (Ambiente B)
    surfaces = EnvironmentLibrary.get_env_B_rectangular(w=10, h=10, ref=0.5, rec_y=4)
    
    rays, metrics = run_simulation(source, surfaces, max_bounces=10)
    
    # Comprobar Balance Total
    sum_energy = metrics['e_captada'] + metrics['e_absorbida'] + metrics['e_escapada'] + metrics['e_limite']
    assert np.isclose(sum_energy, metrics['e_emitida'])
    
    # Si todo se disparó hacia arriba, muchos rayos serán captados directamente (1 rebote o ninguno)
    assert metrics['e_captada'] > 0

def test_integration_3d_cylindrical():
    # 50 Rayos esféricos dentro de un cilindro perfectamente reflectante
    source = Source3D(position=[0, 0, 1], dir_vec=[0, 0, 1], angle_spread=90, num_rays=50, total_energy=500.0)
    
    # Usando la nueva sintaxis de EnvironmentLibrary (Ambiente A en 3D)
    surfaces = EnvironmentLibrary.get_env_A_cylindrical(radius=5, h=10, ref=1.0, rec_z=8)
    
    rays, metrics = run_simulation(source, surfaces, max_bounces=20)
    
    # Como la reflectividad es 1.0, la absorción debe ser cero rigurosamente.
    # Toda la energía debe estar Captada o Detenida (en vuelo por límite de rebotes).
    assert np.isclose(metrics['e_absorbida'], 0.0)
    assert np.isclose(metrics['e_emitida'], metrics['e_captada'] + metrics['e_limite'] + metrics['e_escapada'])