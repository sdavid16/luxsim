import numpy as np
import plotly.graph_objects as go
from core.ray import RayState


# Paleta semántica coherente entre escenas y dimensiones.
INK = "#24364b"
GRID = "#e8eef5"
WALL = "#8292a8"
WALL_FILL = "rgba(148,163,184,0.12)"
RECEIVER = "#0f9b8e"
ESCAPE = "#f59e0b"
RAY_ACTIVE = "#e6a817"
RAY_CAPTURED = "#0f9b8e"
RAY_ABSORBED = "#e05252"
RAY_ESCAPED = "#94a3b8"
RAY_LIMIT = "#8b5cf6"


def get_color(state):
    return {
        RayState.CAPTURED: RAY_CAPTURED,
        RayState.ABSORBED: RAY_ABSORBED,
        RayState.ESCAPED: RAY_ESCAPED,
        RayState.MAX_BOUNCES: RAY_LIMIT,
    }.get(state, RAY_ACTIVE)


def interpolate_ray_history(history, max_dist):
    """Recorta un historial analítico en la distancia acumulada indicada."""
    if not history:
        return np.empty((0, 2))
    pts = []
    for i, item in enumerate(history):
        pos, _, _, dist = item
        if dist <= max_dist:
            pts.append(pos)
            continue
        if i == 0:
            break
        prev_pos, _, _, prev_dist = history[i - 1]
        segment_dist = dist - prev_dist
        if segment_dist > 1e-12 and max_dist >= prev_dist:
            ratio = np.clip((max_dist - prev_dist) / segment_dist, 0.0, 1.0)
            pts.append(prev_pos + ratio * (pos - prev_pos))
        break
    if not pts:
        return np.empty((0, len(history[0][0])))
    return np.asarray(pts, dtype=float)


def _surface_label(surf):
    if surf.type == "RECEPTOR":
        return "Receptor"
    if surf.type == "ABERTURA":
        return "Abertura / escape"
    return "Superficie"


def _surface_color(surf):
    if surf.type == "RECEPTOR":
        return RECEIVER
    if surf.type == "ABERTURA":
        return ESCAPE
    return WALL


def _add_2d_surface(fig, surf):
    name = type(surf).__name__
    color = _surface_color(surf)
    label = _surface_label(surf)
    if name == "Segment2D":
        fig.add_trace(go.Scatter(
            x=[surf.p1[0], surf.p2[0]], y=[surf.p1[1], surf.p2[1]],
            mode="lines", name=label,
            line=dict(color=color, width=5 if surf.type == "RECEPTOR" else 2.5),
            hovertemplate=f"{label}<extra></extra>",
        ))
    elif name == "Circle2D":
        theta = np.linspace(0, 2 * np.pi, 240)
        x = surf.center[0] + surf.radius * np.cos(theta)
        y = surf.center[1] + surf.radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(color=color, width=2.5),
            fill="toself", fillcolor=WALL_FILL,
            hovertemplate=f"{label}<extra></extra>",
        ))
    elif name == "Ellipse2D":
        theta = np.linspace(0, 2 * np.pi, 240)
        x = surf.center[0] + surf.rx * np.cos(theta)
        y = surf.center[1] + surf.ry * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(color=color, width=2.5),
            fill="toself", fillcolor=WALL_FILL,
            hovertemplate=f"{label}<extra></extra>",
        ))


def _add_3d_surface(fig, surf):
    name = type(surf).__name__
    color = _surface_color(surf)
    label = _surface_label(surf)

    if name == "CylinderWall3D":
        theta = np.linspace(0, 2 * np.pi, 64)
        z = np.linspace(surf.cb[2], surf.cb[2] + surf.h, 12)
        th, zz = np.meshgrid(theta, z)
        x = surf.cb[0] + surf.r * np.cos(th)
        y = surf.cb[1] + surf.r * np.sin(th)
        fig.add_trace(go.Surface(
            x=x, y=y, z=zz, name=label, showscale=False,
            opacity=0.17, colorscale=[[0, color], [1, color]],
            hovertemplate=f"{label}<extra></extra>",
            lighting=dict(ambient=0.8, diffuse=0.35, specular=0.12, roughness=0.9),
        ))
    elif name == "Disk3D":
        theta = np.linspace(0, 2 * np.pi, 72)
        radii = np.linspace(0, surf.radius, 14)
        rr, tt = np.meshgrid(radii, theta)
        normal = np.asarray(surf.normal, dtype=float)
        reference = np.array([1., 0., 0.]) if abs(normal[0]) < 0.9 else np.array([0., 1., 0.])
        u = np.cross(normal, reference)
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(normal, u)
        center = np.asarray(surf.center)
        xyz = center[:, None, None] + u[:, None, None] * (rr * np.cos(tt))[None, :, :] + v[:, None, None] * (rr * np.sin(tt))[None, :, :]
        fig.add_trace(go.Surface(
            x=xyz[0], y=xyz[1], z=xyz[2], name=label, showscale=False,
            opacity=0.78 if surf.type == "RECEPTOR" else 0.12,
            colorscale=[[0, color], [1, color]],
            hovertemplate=f"{label}<extra></extra>",
        ))
    elif name == "Ellipsoid3D":
        u = np.linspace(0, 2 * np.pi, 48)
        v = np.linspace(0, np.pi, 28)
        uu, vv = np.meshgrid(u, v)
        x = surf.center[0] + surf.rx * np.cos(uu) * np.sin(vv)
        y = surf.center[1] + surf.ry * np.sin(uu) * np.sin(vv)
        z = surf.center[2] + surf.rz * np.cos(vv)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, name=label, showscale=False,
            opacity=0.16, colorscale=[[0, color], [1, color]],
            hovertemplate=f"{label}<extra></extra>",
            lighting=dict(ambient=0.8, diffuse=0.4, specular=0.1, roughness=0.9),
        ))


def plot_scene(surfaces, rays, is_3d=False, max_time_dist=float("inf")):
    """Dibuja las geometrías implementadas y los historiales reales de los rayos."""
    fig = go.Figure()

    for surf in surfaces:
        if is_3d:
            _add_3d_surface(fig, surf)
        else:
            _add_2d_surface(fig, surf)

    grouped = {}
    for ray in rays:
        pts = interpolate_ray_history(ray.history, max_time_dist)
        if len(pts) < 2:
            continue
        color = get_color(ray.state)
        label = {
            RayState.CAPTURED: "Rayo captado",
            RayState.ABSORBED: "Rayo absorbido",
            RayState.ESCAPED: "Rayo escapado",
            RayState.MAX_BOUNCES: "Límite de rebotes",
            RayState.ACTIVE: "Rayo",
        }.get(ray.state, "Rayo")
        grouped.setdefault((color, label), []).append(pts)

    for (color, label), paths in grouped.items():
        if is_3d:
            xs, ys, zs = [], [], []
            for pts in paths:
                xs.extend(pts[:, 0].tolist() + [None])
                ys.extend(pts[:, 1].tolist() + [None])
                zs.extend(pts[:, 2].tolist() + [None])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name=label,
                line=dict(color=color, width=3), hoverinfo="skip", connectgaps=False,
            ))
        else:
            xs, ys = [], []
            for pts in paths:
                xs.extend(pts[:, 0].tolist() + [None])
                ys.extend(pts[:, 1].tolist() + [None])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=label,
                line=dict(color=color, width=1.6), hoverinfo="skip", connectgaps=False,
            ))

    origins = [np.asarray(ray.history[0][0], dtype=float) for ray in rays if ray.history]
    if origins:
        origin = origins[0]
        if is_3d and len(origin) >= 3:
            fig.add_trace(go.Scatter3d(
                x=[origin[0]], y=[origin[1]], z=[origin[2]],
                mode="markers", name="Fuente",
                marker=dict(size=6, color="#f59e0b", symbol="diamond",
                            line=dict(color="#ffffff", width=1)),
                hovertemplate="Fuente luminosa<extra></extra>",
            ))
        elif not is_3d:
            fig.add_trace(go.Scatter(
                x=[origin[0]], y=[origin[1]], mode="markers", name="Fuente",
                marker=dict(size=11, color="#f59e0b", symbol="diamond",
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate="Fuente luminosa<extra></extra>",
            ))

    # Trazas ficticias transparentes para una leyenda consistente incluso con superficies 3D.
    present_types = {surf.type for surf in surfaces}
    legend_items = [
        ("WALL", WALL, "Superficie"),
        ("RECEPTOR", RECEIVER, "Receptor"),
        ("ABERTURA", ESCAPE, "Abertura / escape"),
    ]
    for surf_type, color, label in legend_items:
        if surf_type in present_types:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines", name=label,
                line=dict(color=color, width=4), showlegend=True,
                visible="legendonly" if is_3d else True,
            ))

    base_layout = dict(
        template="plotly_white",
        height=650 if is_3d else 610,
        margin=dict(l=12, r=12, t=42, b=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfdff",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(255,255,255,0.82)",
            borderwidth=0, font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dbe4ee", font_color=INK),
        uirevision="luxsim-scene",
    )

    if is_3d:
        fig.update_layout(
            **base_layout,
            scene=dict(
                bgcolor="#fbfdff",
                aspectmode="data",
                xaxis=dict(title="X", backgroundcolor="#f8fafc", gridcolor=GRID, zerolinecolor="#cbd5e1"),
                yaxis=dict(title="Y", backgroundcolor="#f8fafc", gridcolor=GRID, zerolinecolor="#cbd5e1"),
                zaxis=dict(title="Z", backgroundcolor="#f8fafc", gridcolor=GRID, zerolinecolor="#cbd5e1"),
                camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
            ),
        )
    else:
        fig.update_layout(
            **base_layout,
            xaxis=dict(title="X", showgrid=True, gridcolor=GRID, zerolinecolor="#cbd5e1", showline=True, linecolor="#cbd5e1"),
            yaxis=dict(title="Y", showgrid=True, gridcolor=GRID, zerolinecolor="#cbd5e1", showline=True, linecolor="#cbd5e1", scaleanchor="x", scaleratio=1),
        )
    return fig
