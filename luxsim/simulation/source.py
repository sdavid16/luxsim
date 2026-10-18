import numpy as np
from core.ray import Ray
from core.vectors import normalize

class Source2D:
    def __init__(self, position, angle_center, angle_spread, num_rays, total_energy, seed=None):
        self.position = np.array(position, dtype=float)
        self.angle_center = np.radians(angle_center)
        self.angle_spread = np.radians(angle_spread)
        self.num_rays = num_rays
        self.total_energy = total_energy
        if seed is not None:
            np.random.seed(seed)

    def generate_rays(self):
        rays = []
        energy_per_ray = self.total_energy / self.num_rays
        
        if self.num_rays == 1:
            angles = [self.angle_center]
        else:
            angles = np.linspace(
                self.angle_center - self.angle_spread/2,
                self.angle_center + self.angle_spread/2, 
                self.num_rays
            )
            
        for ang in angles:
            direction = [np.cos(ang), np.sin(ang)]
            rays.append(Ray(self.position, direction, energy=energy_per_ray))
        return rays

class Source3D:
    def __init__(self, position, dir_vec, angle_spread, num_rays, total_energy, seed=None):
        self.position = np.array(position, dtype=float)
        self.dir_vec = normalize(np.array(dir_vec, dtype=float))
        self.angle_spread = np.radians(angle_spread)
        self.num_rays = num_rays
        self.total_energy = total_energy
        if seed is not None:
            np.random.seed(seed)

    def generate_rays(self):
        rays = []
        energy_per_ray = self.total_energy / self.num_rays
        
        # Sistema de coordenadas local para el cono de dispersión
        up = np.array([0, 0, 1]) if abs(self.dir_vec[2]) < 0.99 else np.array([1, 0, 0])
        u = normalize(np.cross(up, self.dir_vec))
        v = np.cross(self.dir_vec, u)

        # Espiral de Fibonacci para distribuir rayos uniformente en un casquete esférico
        max_angle = self.angle_spread / 2.0
        for i in range(self.num_rays):
            if self.num_rays == 1:
                d = self.dir_vec
            else:
                z = 1 - (i / (self.num_rays - 1)) * (1 - np.cos(max_angle))
                theta = np.sqrt(self.num_rays * np.pi) * np.arccos(z)
                r = np.sqrt(1 - z*z)
                d = (r * np.cos(theta)) * u + (r * np.sin(theta)) * v + z * self.dir_vec
            rays.append(Ray(self.position, d, energy=energy_per_ray))
        return rays