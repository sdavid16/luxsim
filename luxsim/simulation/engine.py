from core.intersections import trace_ray
from core.ray import RayState

def run_simulation(source, surfaces, max_bounces=50, epsilon=1e-7):
    rays = source.generate_rays()
    
    for ray in rays:
        ray.max_bounces = max_bounces
        trace_ray(ray, surfaces, epsilon)
        
    # Balance Termodinámico Estricto
    e_emitida = source.total_energy
    e_captada = sum(r.energy for r in rays if r.state == RayState.CAPTURED)
    e_escapada = sum(r.energy for r in rays if r.state == RayState.ESCAPED)
    e_limite = sum(r.energy for r in rays if r.state == RayState.MAX_BOUNCES)
    
    # La energía absorbida por las paredes es el remanente (Conservación de Energía)
    e_absorbida = e_emitida - e_captada - e_escapada - e_limite

    metrics = {
        "e_emitida": e_emitida,
        "e_captada": e_captada,
        "e_absorbida": e_absorbida,
        "e_escapada": e_escapada,
        "e_limite": e_limite,
        "eficiencia_pct": (e_captada / e_emitida) * 100 if e_emitida > 0 else 0,
        "rayos_captados_pct": (sum(1 for r in rays if r.state == RayState.CAPTURED) / len(rays)) * 100 if len(rays) > 0 else 0,
        "rebotes_medios": sum(r.bounces for r in rays) / len(rays) if len(rays) > 0 else 0
    }
    
    return rays, metrics