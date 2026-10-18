import numpy as np
from .surfaces import SurfaceType
from .vectors import reflect
from .ray import RayState

def calculate_closest_intersection(ray, surfaces, epsilon=1e-7):
    """Encuentra la superficie más cercana intersectada por el rayo."""
    min_t = float('inf')
    best_surface = None
    best_normal = None
    
    for surface in surfaces:
        result = surface.intersect(ray.origin, ray.direction, epsilon)
        if result:
            t, normal = result
            if t < min_t:
                min_t = t
                best_surface = surface
                best_normal = normal
                
    return min_t, best_surface, best_normal

def step_ray(ray, surfaces, epsilon=1e-7):
    if ray.state != RayState.ACTIVE:
        return
        
    t, surface, normal = calculate_closest_intersection(ray, surfaces, epsilon)
    
    if not surface:
        ray.record_state(ray.origin + ray.direction * 100.0, RayState.ESCAPED)
        return
        
    hit_point = ray.origin + t * ray.direction
    
    if surface.type == SurfaceType.ESCAPE:
        ray.origin = hit_point
        ray.record_state(hit_point, RayState.ESCAPED)
        return
        
    if surface.type == SurfaceType.RECEIVER:
        ray.origin = hit_point
        ray.captured_by = surface
        ray.record_state(hit_point, RayState.CAPTURED)
        return
        
    ray.bounces += 1
    new_energy = ray.energy * surface.reflectivity
    ray.origin = hit_point
    ray.direction = reflect(ray.direction, normal)
    ray.energy = new_energy
    
    if ray.energy < 1e-6 or surface.reflectivity == 0.0:
        ray.energy = 0.0
        ray.record_state(hit_point, RayState.ABSORBED)
    elif ray.bounces >= ray.max_bounces:
        ray.record_state(hit_point, RayState.MAX_BOUNCES)
    else:
        ray.record_state(hit_point, RayState.ACTIVE)

def trace_ray(ray, surfaces, epsilon=1e-7):
    """Ejecuta el trazado completo del rayo hasta que deja de estar ACTIVO."""
    while ray.state == RayState.ACTIVE:
        step_ray(ray, surfaces, epsilon)