"""Fuentes direccionales 2D y 3D que distribuyen energía entre rayos."""

import numpy as np
from core.ray import Ray
from core.vectors import normalize

class Source2D:
    """Fuente 2D con dirección central y apertura angular uniforme."""
    def __init__(self, position, angle_center, angle_spread, num_rays, total_energy, seed=None):
        self.position = np.array(position, dtype=float)
        self.angle_center = np.radians(angle_center)
        self.angle_spread = np.radians(angle_spread)
        self.num_rays = num_rays
        self.total_energy = total_energy
        if seed is not None:
            np.random.seed(seed)

    def generate_rays(self):
        """Distribuye los rayos en el abanico y reparte la energía total."""
        rays = []
        energy_per_ray = self.total_energy / self.num_rays
        
        if self.num_rays == 1:
            angles = [self.angle_center]
        else:
            full_circle = np.isclose(self.angle_spread, 2.0 * np.pi)
            angles = np.linspace(
                self.angle_center - self.angle_spread / 2.0,
                self.angle_center + self.angle_spread / 2.0,
                self.num_rays,
                endpoint=not full_circle,
            )
            
        for ang in angles:
            direction = [np.cos(ang), np.sin(ang)]
            rays.append(Ray(self.position, direction, energy=energy_per_ray))
        return rays

class Source3D:
    """Fuente 3D con apertura cónica; 360° distribuye rayos por toda la esfera."""
    def __init__(self, position, dir_vec, angle_spread, num_rays, total_energy, seed=None):
        self.position = np.array(position, dtype=float)
        self.dir_vec = normalize(np.array(dir_vec, dtype=float))
        self.angle_spread = np.radians(angle_spread)
        self.num_rays = num_rays
        self.total_energy = total_energy
        if seed is not None:
            np.random.seed(seed)

    def generate_rays(self):
        """Genera direcciones distribuidas en 3D sin cambiar la energía total."""
        rays = []
        energy_per_ray = self.total_energy / self.num_rays
        
        # Sistema de coordenadas local para el cono de dispersión
        up = np.array([0, 0, 1]) if abs(self.dir_vec[2]) < 0.99 else np.array([1, 0, 0])
        u = normalize(np.cross(up, self.dir_vec))
        v = np.cross(self.dir_vec, u)

        # Distribución de Fibonacci sobre un casquete esférico.
        # El ángulo polar controla la apertura y el azimut distribuye los
        # rayos alrededor del eje, evitando que la emisión parezca casi 2D.
        max_angle = self.angle_spread / 2.0
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(self.num_rays):
            if self.num_rays == 1:
                d = self.dir_vec
            else:
                cos_polar = 1.0 - ((i + 0.5) / self.num_rays) * (1.0 - np.cos(max_angle))
                sin_polar = np.sqrt(max(0.0, 1.0 - cos_polar * cos_polar))
                azimuth = i * golden_angle
                d = (
                    sin_polar * np.cos(azimuth) * u
                    + sin_polar * np.sin(azimuth) * v
                    + cos_polar * self.dir_vec
                )
            rays.append(Ray(self.position, d, energy=energy_per_ray))
        return rays