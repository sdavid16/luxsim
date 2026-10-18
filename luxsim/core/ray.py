"""Estado e historial acumulado de un rayo geométrico."""

import numpy as np
from enum import Enum
from .vectors import normalize

class RayState(Enum):
    ACTIVE = "ACTIVO"
    CAPTURED = "CAPTADO"
    ABSORBED = "ABSORBIDO"
    ESCAPED = "ESCAPADO"
    MAX_BOUNCES = "LIMITE_REBOTES"

class Ray:
    def __init__(self, origin, direction, energy=100.0, max_bounces=50, ray_id=0):
        self.id = ray_id
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))
        self.energy = energy
        self.initial_energy = energy
        self.bounces = 0
        self.max_bounces = max_bounces
        self.state = RayState.ACTIVE
        self.cumulative_distance = 0.0
        self.captured_by = None # Referencia a la superficie que lo captó
        
        # Historial: (posición, energía, estado, distancia_acumulada)
        self.history = [(self.origin.copy(), self.energy, self.state, self.cumulative_distance)]

    def record_state(self, position, state=None):
        if state is not None:
            self.state = state
        dist = np.linalg.norm(position - self.history[-1][0])
        self.cumulative_distance += dist
        self.history.append((position.copy(), self.energy, self.state, self.cumulative_distance))