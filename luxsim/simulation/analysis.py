import numpy as np

def calculate_heatmap_2d(rays, bins=20):
    """Calcula la irradiancia estimada (energía por celda) en el receptor 2D."""
    hits = []
    for r in rays:
        if r.state.name == "CAPTURED" and r.captured_by is not None:
            if type(r.captured_by).__name__ == "Segment2D":
                uv = r.captured_by.get_uv(r.history[-1][0])
                hits.append((uv, r.energy))
                
    if not hits: return np.zeros(bins), 0.0
    
    positions = [h[0] for h in hits]
    energies = [h[1] for h in hits]
    
    hist, bin_edges = np.histogram(positions, bins=bins, weights=energies)
    mean_e = np.mean(hist)
    cv = (np.std(hist) / mean_e) if mean_e > 0 else 0
    return hist, cv

def grid_search_optimization(source_func, surfaces_func, metric_target, param_range):
    """Busca el valor de un parámetro que maximice una métrica (ej: e_captada)."""
    from simulation.engine import run_simulation
    best_val = None
    best_metric = -float('inf')
    
    for val in param_range:
        source = source_func(val)
        surfaces = surfaces_func()
        rays, metrics = run_simulation(source, surfaces)
        
        target_val = metrics[metric_target]
        # Si buscamos reducir rebotes, invertimos el signo
        if metric_target == 'rebotes_medios': target_val = -target_val 
            
        if target_val > best_metric:
            best_metric = target_val
            best_val = val
            
    return best_val, abs(best_metric)