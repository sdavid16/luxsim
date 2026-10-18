import plotly.graph_objects as go
import numpy as np
from core.ray import RayState

def get_color(state):
    return {RayState.CAPTURED: 'green', RayState.ABSORBED: 'red', 
            RayState.ESCAPED: 'gray', RayState.MAX_BOUNCES: 'orange'}.get(state, 'blue')

def interpolate_ray_history(history, max_dist):
    """Trunca el historial geométrico del rayo hasta una distancia máxima (Para Animación)."""
    pts = []
    for i in range(len(history)):
        pos, _, _, dist = history[i]
        if dist <= max_dist:
            pts.append(pos)
        else:
            # Interpolar el último segmento visible
            prev_pos, _, _, prev_dist = history[i-1]
            ratio = (max_dist - prev_dist) / (dist - prev_dist)
            interp_pos = prev_pos + ratio * (pos - prev_pos)
            pts.append(interp_pos)
            break
    return np.array(pts) if pts else np.empty((0, len(history[0][0])))

def plot_scene(surfaces, rays, is_3d=False, max_time_dist=float('inf')):
    fig = go.Figure()
    
    for surf in surfaces:
        c = 'purple' if surf.type == 'RECEPTOR' else ('yellow' if surf.type == 'ABERTURA' else 'blue')
        # ... (Renderizado de geometrías omitido por brevedad, igual al anterior pero admite Elipses) ...
        if type(surf).__name__ == "Ellipse2D" and not is_3d:
            th = np.linspace(0, 2*np.pi, 100)
            x, y = surf.center[0] + surf.rx*np.cos(th), surf.center[1] + surf.ry*np.sin(th)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=c)))
        elif type(surf).__name__ == "Segment2D" and not is_3d:
            fig.add_trace(go.Scatter(x=[surf.p1[0], surf.p2[0]], y=[surf.p1[1], surf.p2[1]], mode='lines', line=dict(color=c, width=3)))
            
    for r in rays:
        pts = interpolate_ray_history(r.history, max_time_dist)
        if len(pts) > 1:
            if is_3d:
                fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines', line=dict(color=get_color(r.state), width=2)))
            else:
                fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='lines', line=dict(color=get_color(r.state), width=1.5)))

    fig.update_layout(showlegend=False, uirevision='const', margin=dict(l=0,r=0,t=30,b=0))
    if not is_3d: fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig