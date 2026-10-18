import time

import numpy as np
import streamlit as st
from core.ray import RayState
from core.surfaces import Segment2D, SurfaceType
from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation
from simulation.source import Source2D, Source3D
from visualization.plotter import plot_scene

st.set_page_config(
    page_title="LUXSIM | Propagación geométrica",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#e5eefc; --muted:#9fb2cc; --line:#263750; --accent:#38bdf8; }
      .stApp { background:radial-gradient(ellipse at 70% 0%,#142b49 0%,#08111f 58%); color:var(--ink); }
      .block-container { max-width:1700px; padding-top:1rem; padding-bottom:1.5rem; }
      [data-testid="stSidebar"] { background:#0b1728; border-right:1px solid #233650; }
      [data-testid="stSidebar"] * { color:#dce8f8; }
      .lux-brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.35rem; }
      .lux-mark { display:flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:14px; background:linear-gradient(135deg,#06b6d4,#6366f1 55%,#a855f7); color:white; font-size:24px; font-weight:800; box-shadow:0 0 26px #06b6d455; }
      .lux-title { font-size:2.15rem; font-weight:850; letter-spacing:-.05em; color:#f8fbff; line-height:1.1; }
      .lux-subtitle { color:var(--muted); font-size:.96rem; margin:.35rem 0 1rem; }
      .lux-eyebrow { color:#67e8f9; text-transform:uppercase; letter-spacing:.16em; font-size:.7rem; font-weight:800; }
      .lux-section { font-size:1.08rem; font-weight:750; color:#e8f2ff; margin:.15rem 0; }
      div[data-testid="stMetric"] { background:#101f33; border:1px solid #29405e; border-radius:14px; padding:10px 14px; }
      div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#e8f2ff; }
      .stButton > button { border-radius:11px; font-weight:700; border:1px solid #36506f; }
      .stButton > button[kind="primary"] { background:linear-gradient(100deg,#0891b2,#6366f1); border:0; color:white; box-shadow:0 5px 20px #0891b233; }
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#101f33; }
      hr { border-color:#263750; }
    </style>
    """,
    unsafe_allow_html=True,
)

ENVIRONMENTS_2D = {"A": "Circunferencia", "B": "Elipse", "C": "Recta reflectante"}
ENVIRONMENTS_3D = {"A": "Cilindro", "B": "Esfera", "C": "Elipsoide"}
MAX_RAYS = 10
MAX_BOUNCES = 12


def build_environment(env_id, mode):
    if mode == "3D":
        if env_id == "A":
            scene = EnvironmentLibrary.get_env_A_cylindrical(5, 10, 1.0, 8)
        elif env_id == "B":
            scene = EnvironmentLibrary.get_env_C_ellipsoid(5, 5, 5, 1.0, 2)
        elif env_id == "C":
            scene = EnvironmentLibrary.get_env_C_ellipsoid(6, 4, 8, 1.0, 2)
        else:
            raise ValueError("Escena 3D no disponible.")
    else:
        if env_id == "A":
            scene = EnvironmentLibrary.get_env_A_circular(5, 1.0, 3)
        elif env_id == "B":
            scene = EnvironmentLibrary.get_env_C_elliptical(6, 4, 1.0, 2.5)
        elif env_id == "C":
            scene = [Segment2D([-8.0, 0.0], [8.0, 0.0], 1.0)]
        else:
            raise ValueError("Escena 2D no disponible.")
    return [surface for surface in scene if surface.type != SurfaceType.RECEIVER]


def is_inside_figure(env_id, mode, position):
    x, y = position[:2]
    if mode == "2D":
        if env_id == "A":
            return x * x + y * y < 25.0 - 1e-8
        if env_id == "B":
            return (x / 6.0) ** 2 + (y / 4.0) ** 2 < 1.0 - 1e-8
        return False
    z = position[2]
    if env_id == "A":
        return x * x + y * y < 25.0 - 1e-8 and 1e-8 < z < 10.0 - 1e-8
    if env_id == "B":
        return x * x + y * y + z * z < 25.0 - 1e-8
    if env_id == "C":
        return (x / 6.0) ** 2 + (y / 4.0) ** 2 + (z / 8.0) ** 2 < 1.0 - 1e-8
    return False


def build_source(mode, position, angle_2d, azimuth_3d, elevation_3d, aperture, ray_count):
    if mode == "2D":
        return Source2D(position[:2], angle_2d, aperture, ray_count, 1000.0)

    azimuth = np.radians(azimuth_3d)
    elevation = np.radians(elevation_3d)
    direction = [
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ]
    return Source3D(position[:3], direction, aperture, ray_count, 1000.0)


FORMULAS = {
    ("2D", "A"): ("Circunferencia", r"(x-h)^2+(y-k)^2=r^2"),
    ("2D", "B"): ("Elipse", r"\frac{(x-h)^2}{a^2}+\frac{(y-k)^2}{b^2}=1"),
    ("2D", "C"): ("Recta reflectante", r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d}"),
    ("3D", "A"): ("Cilindro", r"(x-h)^2+(y-k)^2=r^2,\quad z_{\min}\leq z\leq z_{\max}"),
    ("3D", "B"): ("Esfera", r"(x-h)^2+(y-k)^2+(z-l)^2=r^2"),
    ("3D", "C"): ("Elipsoide", r"\frac{(x-h)^2}{a^2}+\frac{(y-k)^2}{b^2}+\frac{(z-l)^2}{c^2}=1"),
}


def show_formulas(mode, env_id, angle_2d, azimuth_3d, elevation_3d):
    left, right = st.columns(2)
    figure_name, figure_formula = FORMULAS[(mode, env_id)]
    with left:
        st.markdown("**1. Figura geométrica**")
        st.caption(figure_name)
        st.latex(figure_formula)
        st.markdown("**2. Trayectoria del rayo**")
        if mode == "2D":
            st.latex(r"\mathbf{d}=(\cos\theta,\sin\theta)")
            st.caption(f"Dirección central actual: {angle_2d}°")
        else:
            st.latex(r"\mathbf{d}=(\cos\varphi\cos\theta,\cos\varphi\sin\theta,\sin\varphi)")
            st.caption(f"Azimut: {azimuth_3d}° · elevación: {elevation_3d}°")
        st.latex(r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d},\quad t\geq0")
    with right:
        st.markdown("**3. Intersección y rebote**")
        st.caption("Se busca el primer punto de la superficie que cumple la ecuación de la trayectoria.")
        st.latex(r"\mathbf{p}_i=\mathbf{p}_0+t_i\mathbf{d}")
        st.caption("En un rebote especular ideal, la componente normal de la dirección cambia de signo.")
        st.latex(r"\mathbf{d}_{\mathrm{salida}}=\mathbf{d}_{\mathrm{entrada}}-2(\mathbf{d}_{\mathrm{entrada}}\cdot\mathbf{n})\mathbf{n}")
        st.caption("Aquí n es la normal unitaria a la superficie en el punto de impacto.")


if "config_signature" not in st.session_state:
    st.session_state.update({
        "rays": [],
        "metrics": {},
        "surfaces": [],
        "max_sim_dist": 1.0,
        "config_signature": None,
        "animation_progress": 0.0,
        "animation_playing": False,
        "animation_last_tick": time.monotonic(),
        "validation_error": None,
        "has_applied_config": False,
    })

with st.sidebar:
    st.markdown(
        '<div class="lux-brand"><div class="lux-mark">✦</div><div><div class="lux-eyebrow">Laboratorio geométrico</div><div style="font-size:1.2rem;font-weight:800;color:#17283f;">LUXSIM</div></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**1 · Escena**")
    with st.form("simulation_settings", border=False):
        mode = st.radio("Dimensión", ["2D", "3D"], horizontal=True, key="ui_mode")
        options = ENVIRONMENTS_2D if mode == "2D" else ENVIRONMENTS_3D
        env_id = st.selectbox("Figura", list(options), format_func=lambda key: options[key], key=f"scene_{mode}")
    
        st.markdown("**2 · Ubicación de la fuente**")
        source_region = st.radio(
            "La fuente estará",
            ["Dentro de la figura", "Fuera de la figura"],
            horizontal=False,
            key=f"source_region_{mode}",
        )
        if mode == "2D":
            source_x = st.slider("Coordenada X", -8.0, 8.0, 0.0, 0.5, key="source_x_2d")
            source_y = st.slider("Coordenada Y", -8.0, 8.0, 2.0, 0.5, key="source_y_2d")
            position = [source_x, source_y]
            angle_2d = st.slider("Dirección central (°)", 0, 355, 270, 5, key="angle_2d")
            aperture = st.slider("Apertura angular (°)", 0, 100, 50, 5, key="aperture_2d")
            azimuth_3d, elevation_3d = 30, 55
        else:
            source_x = st.slider("Coordenada X", -8.0, 8.0, 0.0, 0.5, key="source_x_3d")
            source_y = st.slider("Coordenada Y", -8.0, 8.0, 0.0, 0.5, key="source_y_3d")
            source_z = st.slider("Coordenada Z", -3.0, 13.0, 1.0, 0.5, key="source_z_3d")
            position = [source_x, source_y, source_z]
            azimuth_3d = st.slider("Azimut (°)", 0, 355, 30, 5, key="azimuth_3d")
            elevation_3d = st.slider("Elevación (°)", -85, 85, 25, 5, key="elevation_3d")
            aperture = st.slider("Apertura del cono (°)", 0, 140, 100, 5, key="aperture_3d")
            angle_2d = 90
    
        ray_max = MAX_RAYS
        ray_default = 7
        ray_key = "ray_count_2d" if mode == "2D" else "ray_count_3d"
        ray_count = st.slider("Rayos emitidos", 1, ray_max, ray_default, 1, key=ray_key)
        max_distance = st.slider("Alcance máximo (unidades)", 5, 30, 15, 5, key="max_ray_distance")
        submitted = st.form_submit_button("Aplicar parámetros y simular", type="primary", use_container_width=True)

if submitted:
    st.session_state.has_applied_config = True

signature = (mode, env_id, source_region, tuple(position), angle_2d, azimuth_3d, elevation_3d, aperture, ray_count, max_distance)
inside = is_inside_figure(env_id, mode, position)
placement_ok = (mode == "2D" and env_id == "C") or (inside if source_region == "Dentro de la figura" else not inside)

# Al cambiar posición, orientación o escena se recalcula automáticamente.
# Así la fuente puede desplazarse sin tener que iniciar de nuevo manualmente.
if submitted:
    if placement_ok:
        try:
            scene = build_environment(env_id, mode)
            source = build_source(mode, position, angle_2d, azimuth_3d, elevation_3d, aperture, ray_count)
            rays, metrics = run_simulation(source, scene, MAX_BOUNCES, max_distance=float(max_distance))
            max_dist = float(max_distance)
            st.session_state.update({
                "rays": rays,
                "metrics": metrics,
                "surfaces": scene,
                "max_sim_dist": max(max_dist, 0.01),
                "config_signature": signature,
                "animation_progress": 0.0,
                "animation_playing": True,
                "animation_last_tick": time.monotonic(),
                "validation_error": None,
            })
        except Exception as exc:
            st.session_state.validation_error = f"No se pudo calcular la escena: {exc}"
    else:
        st.session_state.config_signature = None
        st.session_state.animation_playing = False
        st.session_state.validation_error = (
            "La fuente no está en la región elegida. Ajusta sus coordenadas o cambia entre interior y exterior."
        )

st.markdown(
    '<div class="lux-eyebrow">GEOMETRÍA ANALÍTICA · TRAZADO DE RAYOS</div>'
    '<div class="lux-title">LUXSIM <span style="color:#38bdf8;">/</span> Propagación y rebote</div>',
    unsafe_allow_html=True,
)

if not st.session_state.has_applied_config:
    preview_scene = build_environment(env_id, mode)
    preview = plot_scene(preview_scene, [], is_3d=(mode == "3D"), source_position=position,
        beam_direction=(np.cos(np.radians(angle_2d)), np.sin(np.radians(angle_2d))) if mode == "2D" else None,
        beam_direction_3d=(np.cos(np.radians(elevation_3d)) * np.cos(np.radians(azimuth_3d)), np.cos(np.radians(elevation_3d)) * np.sin(np.radians(azimuth_3d)), np.sin(np.radians(elevation_3d))) if mode == "3D" else None,
        aperture_degrees=aperture, beam_length=float(max_distance), ray_count=ray_count)
    st.plotly_chart(preview, use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "responsive": True})
elif st.session_state.validation_error:
    st.error(st.session_state.validation_error)
    preview_scene = build_environment(env_id, mode)
    preview = plot_scene(
        preview_scene,
        [],
        is_3d=(mode == "3D"),
        source_position=position,
        beam_direction=(np.cos(np.radians(angle_2d)), np.sin(np.radians(angle_2d))) if mode == "2D" else None,
        beam_direction_3d=(np.cos(np.radians(elevation_3d)) * np.cos(np.radians(azimuth_3d)), np.cos(np.radians(elevation_3d)) * np.sin(np.radians(azimuth_3d)), np.sin(np.radians(elevation_3d))) if mode == "3D" else None,
        aperture_degrees=aperture, beam_length=float(max_distance), ray_count=ray_count,
    )
    st.plotly_chart(
        preview,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        key=f"source_preview_{mode}_{env_id}",
    )
else:
    rays = st.session_state.rays
    metrics = st.session_state.metrics
    escaped = sum(ray.state == RayState.ESCAPED for ray in rays)
    capped = sum(ray.state == RayState.MAX_BOUNCES for ray in rays)
    absorbed = sum(ray.state == RayState.ABSORBED for ray in rays)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rayos emitidos", len(rays))
    c2.metric("Rayos que escaparon", escaped)
    c3.metric("Rayos que alcanzaron el límite de rebotes", capped)

    @st.fragment(run_every=0.10 if st.session_state.animation_playing else None)
    def animated_scene():
        controls = st.columns([1, 1, 2, 4])
        if st.session_state.animation_playing:
            if controls[0].button("Pausar", use_container_width=True):
                st.session_state.animation_playing = False
                st.session_state.animation_last_tick = time.monotonic()
                st.rerun()
        else:
            if controls[0].button("Continuar", use_container_width=True):
                st.session_state.animation_playing = True
                st.session_state.animation_last_tick = time.monotonic()
                st.rerun()
        if controls[1].button("Reiniciar", use_container_width=True):
            st.session_state.animation_progress = 0.0
            st.session_state.animation_last_tick = time.monotonic()
            st.session_state.animation_playing = True
            st.rerun()
        speed = controls[2].slider(
            "Velocidad",
            min_value=0.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="animation_speed_fragment",
        )
        controls[3].caption(f"Velocidad de propagación: {speed:.1f} unidades/s")

        if st.session_state.animation_playing:
            now = time.monotonic()
            elapsed = max(0.0, min(now - st.session_state.animation_last_tick, 0.25))
            st.session_state.animation_progress = min(
                st.session_state.max_sim_dist,
                st.session_state.animation_progress + elapsed * float(speed),
            )
            st.session_state.animation_last_tick = now
            if st.session_state.animation_progress >= st.session_state.max_sim_dist:
                st.session_state.animation_playing = False
                st.rerun()

        progress = float(st.session_state.animation_progress)
        st.markdown(f'<div class="lux-section">{options[env_id]} · vista {mode}</div>', unsafe_allow_html=True)
        fig = plot_scene(
            st.session_state.surfaces,
            st.session_state.rays,
            is_3d=(mode == "3D"),
            max_time_dist=progress,
            source_position=position,
            beam_direction=(np.cos(np.radians(angle_2d)), np.sin(np.radians(angle_2d))) if mode == "2D" else None,
            beam_direction_3d=(np.cos(np.radians(elevation_3d)) * np.cos(np.radians(azimuth_3d)), np.cos(np.radians(elevation_3d)) * np.sin(np.radians(azimuth_3d)), np.sin(np.radians(elevation_3d))) if mode == "3D" else None,
            aperture_degrees=aperture, beam_length=float(max_distance), ray_count=ray_count,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            key=f"scene_plot_{mode}_{env_id}",
        )
    
    animated_scene()

    with st.expander("Conceptos y fórmulas", expanded=False):
        show_formulas(mode, env_id, angle_2d, azimuth_3d, elevation_3d)
        st.divider()
