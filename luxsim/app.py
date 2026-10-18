import time

import streamlit as st
from core.ray import RayState
from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation
from simulation.source import Source2D, Source3D
from visualization.plotter import plot_scene


st.set_page_config(
    page_title="LUXSIM | Geometría de rayos",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#17283f; --muted:#64748b; --line:#e2e8f0; --accent:#0f766e; }
      .stApp { background:#f5f8fc; color:var(--ink); }
      .block-container { max-width:1580px; padding-top:1.2rem; padding-bottom:2rem; }
      [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
      .lux-brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.35rem; }
      .lux-mark { display:flex; align-items:center; justify-content:center; width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,#0f766e,#14b8a6); color:white; font-size:22px; font-weight:800; }
      .lux-title { font-size:1.9rem; font-weight:800; letter-spacing:-.05em; color:var(--ink); line-height:1.1; }
      .lux-subtitle { color:var(--muted); font-size:.96rem; margin:.35rem 0 1rem; }
      .lux-eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.13em; font-size:.7rem; font-weight:800; }
      .lux-section { font-size:1.08rem; font-weight:750; color:var(--ink); margin:.15rem 0; }
      .lux-muted { color:var(--muted); font-size:.86rem; }
      div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 15px; }
      div[data-testid="stMetricLabel"] { color:var(--muted); font-size:.82rem; }
      .stButton > button { border-radius:10px; font-weight:650; }
      .stButton > button[kind="primary"] { background:#0f766e; border:0; }
      hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Escenas deliberadamente limitadas a las geometrías más útiles para el curso.
ENVIRONMENTS_2D = {
    "A": "Circunferencia",
    "B": "Rectángulo",
    "C": "Elipse",
}
ENVIRONMENTS_3D = {
    "A": "Cilindro",
    "C": "Elipsoide",
}
RAYS_2D = 7
RAYS_3D = 19
MAX_BOUNCES = 12


def build_environment(env_id, mode):
    """Escenas ideales: paredes perfectamente reflectantes y receptor definido."""
    if mode == "3D":
        if env_id == "A":
            return EnvironmentLibrary.get_env_A_cylindrical(5, 10, 1.0, 8)
        if env_id == "C":
            return EnvironmentLibrary.get_env_C_ellipsoid(6, 4, 8, 1.0, 2)
    else:
        if env_id == "A":
            return EnvironmentLibrary.get_env_A_circular(5, 1.0, 3)
        if env_id == "B":
            return EnvironmentLibrary.get_env_B_rectangular(10, 10, 1.0, 3.5)
        if env_id == "C":
            return EnvironmentLibrary.get_env_C_elliptical(6, 4, 1.0, 2.5)
    raise ValueError("La escena seleccionada no está disponible en esta dimensión.")


def build_source(env_id, mode, angle, aperture):
    """Cantidad de rayos fija; los únicos parámetros son geométricos."""
    if mode == "3D":
        # Eje oblicuo y apertura amplia para hacer visible la dispersión XYZ.
        direction = [0.48, 0.24, 0.84] if env_id == "A" else [0.42, 0.28, 0.86]
        position = [0, 0, 1] if env_id == "A" else [0, 0, 0]
        return Source3D(position, direction, aperture, RAYS_3D, 1000.0)

    presets = {
        "A": ([0, -2], 90),
        "B": ([0, -2], 90),
        "C": ([0, -1.5], 90),
    }
    position, default_angle = presets[env_id]
    center_angle = default_angle if angle is None else angle
    return Source2D(position, center_angle, aperture, RAYS_2D, 1000.0)


FORMULAS = {
    ("2D", "A"): [
        ("Circunferencia (lugar geométrico)", r"(x-h)^2+(y-k)^2=r^2"),
        ("Recta que representa cada rayo", r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d},\quad t\geq 0"),
    ],
    ("2D", "B"): [
        ("Lados del rectángulo", r"\mathbf{p}(u)=\mathbf{a}+u(\mathbf{b}-\mathbf{a}),\quad 0\leq u\leq 1"),
        ("Intersección recta-segmento", r"\mathbf{p}_0+t\mathbf{d}=\mathbf{a}+u(\mathbf{b}-\mathbf{a})"),
    ],
    ("2D", "C"): [
        ("Elipse (lugar geométrico)", r"\frac{(x-h)^2}{a^2}+\frac{(y-k)^2}{b^2}=1"),
        ("Recta que representa cada rayo", r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d},\quad t\geq 0"),
    ],
    ("3D", "A"): [
        ("Superficie cilíndrica", r"(x-h)^2+(y-k)^2=r^2"),
        ("Intervalo de altura", r"z_{\min}\leq z\leq z_{\max}"),
        ("Recta en el espacio", r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d},\quad t\geq 0"),
    ],
    ("3D", "C"): [
        ("Elipsoide", r"\frac{(x-h)^2}{a^2}+\frac{(y-k)^2}{b^2}+\frac{(z-l)^2}{c^2}=1"),
        ("Recta en el espacio", r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d},\quad t\geq 0"),
    ],
}


def show_formulas(mode, env_id):
    st.markdown("**Fórmulas usadas en esta escena**")
    for label, formula in FORMULAS[(mode, env_id)]:
        st.caption(label)
        st.latex(formula)
    st.caption("Intersección: se sustituye la ecuación del rayo en la de la superficie y se elige el primer impacto válido (t > 0).")
    st.latex(r"\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d}")
    st.caption("Reflexión ideal en una superficie con normal unitaria n.")
    st.latex(r"\mathbf{d}'=\mathbf{d}-2(\mathbf{d}\cdot\mathbf{n})\mathbf{n}")


if "rays" not in st.session_state:
    st.session_state.update({
        "rays": [],
        "metrics": {},
        "surfaces": [],
        "max_sim_dist": 1.0,
        "calc_done": False,
        "config_signature": None,
        "animation_progress": 0.0,
        "animation_playing": False,
        "animation_last_tick": time.monotonic(),
    })

with st.sidebar:
    st.markdown(
        '<div class="lux-brand"><div class="lux-mark">✦</div><div><div class="lux-eyebrow">Laboratorio geométrico</div><div style="font-size:1.2rem;font-weight:800;color:#17283f;">LUXSIM</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Rectas, lugares geométricos e intersecciones")
    st.divider()
    st.markdown("**1 · Escena geométrica**")
    mode = st.radio("Dimensión", ["2D", "3D"], horizontal=True, key="ui_mode")
    options = ENVIRONMENTS_2D if mode == "2D" else ENVIRONMENTS_3D
    env_id = st.selectbox(
        "Superficie",
        list(options),
        format_func=lambda key: options[key],
        key=f"scene_{mode}",
    )

    if mode == "2D":
        with st.expander("2 · Dirección del haz", expanded=True):
            angle = st.slider("Dirección central (°)", 0, 360, 90, 5)
            aperture = st.slider("Apertura angular (°)", 0, 100, 50, 5)
        ray_count = RAYS_2D
    else:
        angle = None
        with st.expander("2 · Dirección del haz", expanded=True):
            aperture = st.slider("Apertura del cono (°)", 20, 140, 100, 5)
            st.caption("El eje está inclinado y los rayos se distribuyen en el espacio XYZ.")
        ray_count = RAYS_3D

    st.caption(f"Rayos por escena: **{ray_count}** (cantidad fija).")
    st.caption("Se usa reflexión ideal: no hay refracción ni controles de reflectividad.")
    speed = st.slider(
        "Velocidad de animación (unidades geométricas/s)",
        min_value=0.5,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Controla la velocidad visual del recorrido, no una velocidad física de la luz.",
        key="animation_speed",
    )
    run_clicked = st.button("Iniciar simulación", type="primary", use_container_width=True)

signature = (mode, env_id, angle, aperture)
if run_clicked:
    try:
        scene = build_environment(env_id, mode)
        source = build_source(env_id, mode, angle, aperture)
        rays, metrics = run_simulation(source, scene, MAX_BOUNCES)
        max_dist = max((float(ray.cumulative_distance) for ray in rays), default=1.0)
        st.session_state.update({
            "rays": rays,
            "metrics": metrics,
            "surfaces": scene,
            "max_sim_dist": max(max_dist, 0.01),
            "calc_done": True,
            "config_signature": signature,
            "animation_progress": 0.0,
            "animation_playing": True,
            "animation_last_tick": time.monotonic(),
        })
    except Exception as exc:
        st.error(f"No se pudo calcular la escena: {exc}")

current = st.session_state.calc_done and st.session_state.config_signature == signature

st.markdown(
    '<div class="lux-eyebrow">Geometría analítica aplicada a óptica</div>'
    '<div class="lux-title">LUXSIM <span style="color:#0f766e;">/</span> Trazado de rayos</div>'
    '<div class="lux-subtitle">Observa cómo una recta intersecta una superficie y cómo cambia su dirección al reflejarse. La geometría es el centro de la simulación.</div>',
    unsafe_allow_html=True,
)

if not current:
    st.info("Elige una superficie y pulsa **Iniciar simulación**. El recorrido comenzará automáticamente con la velocidad indicada.")
elif current:
    rays = st.session_state.rays
    metrics = st.session_state.metrics
    captured = sum(ray.state == RayState.CAPTURED for ray in rays)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rayos trazados", f"{len(rays)}", "cantidad fija")
    c2.metric("Impactos en receptor", f"{captured}")
    c3.metric("Rebotes promedio", f"{metrics['rebotes_medios']:.1f}")

    controls = st.columns([1, 1, 4])
    if st.session_state.animation_playing:
        if controls[0].button("Pausar", use_container_width=True):
            st.session_state.animation_playing = False
            st.rerun()
    else:
        if controls[0].button("Continuar", use_container_width=True):
            st.session_state.animation_playing = True
            st.session_state.animation_last_tick = time.monotonic()
            st.rerun()
    if controls[1].button("Reiniciar recorrido", use_container_width=True):
        st.session_state.animation_progress = 0.0
        st.session_state.animation_last_tick = time.monotonic()
        st.session_state.animation_playing = True
        st.rerun()
    controls[2].caption(f"Velocidad: {speed:.1f} unidades geométricas/s · el control de velocidad está en el panel lateral.")

    max_dist = float(st.session_state.max_sim_dist)

    @st.fragment(run_every=0.1 if st.session_state.animation_playing else None)
    def animated_scene():
        if st.session_state.animation_playing:
            now = time.monotonic()
            elapsed = max(0.0, now - st.session_state.animation_last_tick)
            st.session_state.animation_progress = min(
                max_dist,
                st.session_state.animation_progress + elapsed * float(st.session_state.animation_speed),
            )
            st.session_state.animation_last_tick = now
            reached_end = st.session_state.animation_progress >= max_dist
            if reached_end:
                st.session_state.animation_playing = False
        else:
            reached_end = False

        progress = float(st.session_state.animation_progress)
        scene_title = options[env_id]
        st.markdown(
            f'<div class="lux-section">{scene_title} · vista {mode}</div>'
            '<div class="lux-muted">La escena se dibuja a partir de las superficies y los historiales de intersección del motor.</div>',
            unsafe_allow_html=True,
        )
        fig = plot_scene(
            st.session_state.surfaces,
            st.session_state.rays,
            is_3d=(mode == "3D"),
            max_time_dist=progress,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
        )
        st.caption(f"Recorrido visual: {progress:.1f} / {max_dist:.1f} unidades geométricas")
        if reached_end:
            st.rerun()

    animated_scene()

    with st.expander("¿Qué fórmulas aparecen aquí y para qué sirven?", expanded=False):
        show_formulas(mode, env_id)

    with st.expander("Cómo leer la escena", expanded=False):
        st.markdown(
            "- **Fuente (rombo ámbar):** punto desde el que salen los rayos.\n"
            "- **Superficie gris:** circunferencia, segmentos, cilindro o elipsoide donde se calculan intersecciones.\n"
            "- **Receptor verde:** segmento o disco que registra los rayos que llegan.\n"
            "- **Rayos:** cada tramo se dibuja desde un punto de origen hasta el siguiente impacto calculado.\n"
            "- **Ejes X, Y, Z:** permiten comprobar la posición y la dirección espacial en 3D."
        )
