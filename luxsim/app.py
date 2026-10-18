import streamlit as st
import numpy as np
import plotly.graph_objects as go

from simulation.source import Source2D, Source3D
from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation
from visualization.plotter import plot_scene
from simulation.analysis import calculate_heatmap_2d


st.set_page_config(
    page_title="LUXSIM | Laboratorio óptico",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Diseño visual: mantiene Streamlit y Plotly, sin dependencias adicionales.
st.markdown(
    """
    <style>
      :root {
        --lux-ink: #17283f;
        --lux-muted: #64748b;
        --lux-line: #e2e8f0;
        --lux-accent: #0f766e;
        --lux-bg: #f5f8fc;
      }
      .stApp { background: var(--lux-bg); color: var(--lux-ink); }
      .block-container { max-width: 1640px; padding-top: 1.35rem; padding-bottom: 2.5rem; }
      [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--lux-line);
      }
      [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
      .lux-brand { display:flex; align-items:center; gap:.75rem; margin-bottom:.25rem; }
      .lux-mark {
        display:flex; align-items:center; justify-content:center;
        width:42px; height:42px; border-radius:13px;
        background:linear-gradient(135deg,#0f766e,#14b8a6);
        color:#ffffff; font-size:23px; font-weight:800;
        box-shadow:0 5px 14px rgba(15,118,110,.18);
      }
      .lux-title { font-size:1.9rem; font-weight:780; letter-spacing:-.055em; color:var(--lux-ink); line-height:1.05; }
      .lux-subtitle { color:var(--lux-muted); font-size:.96rem; margin:.35rem 0 1.15rem 0; }
      .lux-eyebrow { color:var(--lux-accent); text-transform:uppercase; letter-spacing:.13em; font-size:.7rem; font-weight:750; }
      .lux-section { font-size:1.04rem; font-weight:720; color:var(--lux-ink); margin:.2rem 0 .15rem 0; }
      .lux-muted { color:var(--lux-muted); font-size:.86rem; }
      div[data-testid="stMetric"] {
        background:#ffffff; border:1px solid var(--lux-line); border-radius:13px;
        padding:15px 17px; box-shadow:0 2px 7px rgba(15,23,42,.025);
      }
      div[data-testid="stMetricLabel"] { color:var(--lux-muted); font-size:.82rem; }
      div[data-testid="stMetricValue"] { color:var(--lux-ink); font-weight:750; }
      .stButton > button[kind="primary"] {
        border:0; border-radius:10px; font-weight:700; min-height:2.8rem;
        background:#0f766e;
      }
      .stButton > button[kind="primary"]:hover { background:#115e59; border:0; }
      div[data-testid="stTabs"] button { font-weight:650; }
      div[data-testid="stAlert"] { border-radius:11px; }
      hr { border-color:var(--lux-line); }
    </style>
    """,
    unsafe_allow_html=True,
)

ENVIRONMENTS_2D = {
    "A": "Cámara circular",
    "B": "Recinto rectangular",
    "C": "Cámara elíptica",
    "D": "Galería con abertura",
    "E": "Obstáculo interior",
    "F": "Guía luminosa",
    "G": "Reflectividad mixta",
    "H": "Múltiples receptores",
}
ENVIRONMENTS_3D = {
    "A": "Cámara cilíndrica",
    "C": "Cámara elipsoidal",
}


def build_environment(env_id, mode, reflectivity):
    """Construye una escena con las geometrías analíticas ya implementadas."""
    if mode == "3D":
        if env_id == "A":
            return EnvironmentLibrary.get_env_A_cylindrical(5, 10, reflectivity, 8)
        if env_id == "C":
            return EnvironmentLibrary.get_env_C_ellipsoid(6, 4, 8, reflectivity, 2)
        raise ValueError("Este ambiente no dispone de una geometría 3D implementada.")

    builders = {
        "A": lambda: EnvironmentLibrary.get_env_A_circular(5, reflectivity, 3),
        "B": lambda: EnvironmentLibrary.get_env_B_rectangular(10, 10, reflectivity, 4),
        "C": lambda: EnvironmentLibrary.get_env_C_elliptical(6, 4, reflectivity, 2),
        "D": lambda: EnvironmentLibrary.get_env_D_gallery(10, 10, reflectivity),
        "E": lambda: EnvironmentLibrary.get_env_E_obstacle(10, 10, reflectivity),
        "F": lambda: EnvironmentLibrary.get_env_F_guide(10, 4, reflectivity),
        "G": lambda: EnvironmentLibrary.get_env_G_mixed_ref(10, 10),
        "H": lambda: EnvironmentLibrary.get_env_H_multi_receiver(5, reflectivity),
    }
    return builders[env_id]()


def build_source(env_id, mode, num_rays, spread, angle):
    """Ubica la fuente en una posición inicial válida para cada ambiente."""
    if mode == "3D":
        position = [0, 0, 1] if env_id == "A" else [0, 0, 0]
        return Source3D(position, [0, 0, 1], spread, num_rays, 1000.0)

    presets = {
        "A": ([0, -2], 90),
        "B": ([0, -2], 90),
        "C": ([0, -1.5], 90),
        "D": ([0, 0], 270),
        "E": ([0, -3], 90),
        "F": ([1, 0], 0),
        "G": ([0, -2], 90),
        "H": ([0, 0], 90),
    }
    position, default_angle = presets[env_id]
    return Source2D(position, default_angle if angle is None else angle, spread, num_rays, 1000.0)


if "rays" not in st.session_state:
    st.session_state.update({
        "rays": [],
        "metrics": {},
        "surfaces": [],
        "max_sim_dist": 1.0,
        "calc_done": False,
        "config_signature": None,
        "modo": "2D",
        "env_id": "A",
    })

# Panel de control: los controles quedan agrupados fuera de la zona de visualización.
with st.sidebar:
    st.markdown(
        '<div class="lux-brand"><div class="lux-mark">✦</div><div><div class="lux-eyebrow">Laboratorio óptico</div><div style="font-size:1.18rem;font-weight:780;color:#17283f;">LUXSIM</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Trazado geométrico de rayos y captación de energía")
    st.divider()

    st.markdown("**01 · Escena**")
    mode = st.radio("Dimensionalidad", ["2D", "3D"], horizontal=True, key="ui_mode")
    environment_options = ENVIRONMENTS_2D if mode == "2D" else ENVIRONMENTS_3D
    selected_env = st.selectbox(
        "Ambiente",
        options=list(environment_options.keys()),
        format_func=lambda key: f"{key} · {environment_options[key]}",
        key=f"environment_{mode}",
    )
    if mode == "3D":
        st.caption("En 3D se muestran solo los ambientes con geometría tridimensional implementada.")

    with st.expander("02 · Fuente luminosa", expanded=True):
        n_rays = st.slider("Número de rayos", 1, 200, 48, 1)
        spread = st.slider("Apertura del haz (°)", 1, 180, 60, 1)
        if mode == "2D":
            default_angle = 0 if selected_env == "F" else (270 if selected_env == "D" else 90)
            angle = st.slider(
                "Dirección central (°)",
                0, 360, default_angle, 1,
                key=f"angle_{selected_env}_{mode}",
            )
        else:
            angle = None
            st.caption("La fuente apunta hacia +Z; la apertura controla la dispersión espacial.")

    with st.expander("03 · Superficies y cálculo", expanded=True):
        reflectivity = st.slider("Reflectividad de paredes (ρ)", 0.0, 1.0, 0.82, 0.01)
        max_bounces = st.slider("Límite de rebotes", 1, 100, 25, 1)

    run_clicked = st.button("Calcular simulación", type="primary", use_container_width=True)

current_signature = (
    mode, selected_env, int(n_rays), int(spread), angle,
    round(float(reflectivity), 4), int(max_bounces),
)

if run_clicked:
    try:
        scene = build_environment(selected_env, mode, reflectivity)
        source = build_source(selected_env, mode, n_rays, spread, angle)
        rays, metrics = run_simulation(source, scene, max_bounces)
        max_dist = max((float(ray.cumulative_distance) for ray in rays), default=1.0)
        st.session_state.update({
            "rays": rays,
            "metrics": metrics,
            "surfaces": scene,
            "max_sim_dist": max(max_dist, 0.01),
            "calc_done": True,
            "config_signature": current_signature,
            "modo": mode,
            "env_id": selected_env,
        })
    except Exception as exc:
        st.error(f"No se pudo calcular la escena: {exc}")

results_current = (
    st.session_state.calc_done
    and st.session_state.config_signature == current_signature
)

# Cabecera limpia y visualización como elemento principal.
st.markdown(
    '<div class="lux-eyebrow">Simulación de óptica geométrica</div>'
    '<div class="lux-title">LUXSIM <span style="color:#0f766e;">/</span> Captación de luz</div>'
    '<div class="lux-subtitle">Explora cómo la geometría, la dirección de emisión y la reflectividad afectan la energía que alcanza el receptor.</div>',
    unsafe_allow_html=True,
)

if results_current:
    metrics = st.session_state.metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eficiencia de captación", f"{metrics['eficiencia_pct']:.1f}%")
    m2.metric("Energía captada", f"{metrics['e_captada']:.1f}", "de 1000 unidades")
    m3.metric("Rayos captados", f"{metrics['rayos_captados_pct']:.1f}%")
    m4.metric("Rebotes promedio", f"{metrics['rebotes_medios']:.1f}")

    sim_tab, heat_tab = st.tabs(["✦  Escena y trayectorias", "▦  Distribución de iluminación"])
    with sim_tab:
        scene_name = environment_options[selected_env]
        st.markdown(
            f'<div class="lux-section">{scene_name} · Vista {mode}</div>'
            '<div class="lux-muted">Las superficies y trayectorias proceden de la geometría calculada por el motor.</div>',
            unsafe_allow_html=True,
        )
        max_dist = float(st.session_state.max_sim_dist)
        step = max(max_dist / 300.0, 0.01)
        progress_key = f"ray_progress_{mode}_{selected_env}"
        # Un nuevo cálculo puede tener otra distancia máxima; evita que un valor
        # de slider conservado de una ejecución anterior quede fuera del rango.
        if progress_key in st.session_state:
            st.session_state[progress_key] = min(
                max(float(st.session_state[progress_key]), 0.0), max_dist
            )
        progress = st.slider(
            "Recorrido de los rayos",
            min_value=0.0,
            max_value=max_dist,
            value=max_dist,
            step=step,
            help="Recorta los historiales analíticos para visualizar progresivamente la propagación.",
            key=progress_key,
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
        st.caption(
            f"Vista {mode} · {len(st.session_state.rays)} rayos · "
            f"Progreso de trayectoria: {progress:.2f} / {max_dist:.2f}"
        )

        with st.expander("Balance energético detallado"):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Emitida", f"{metrics['e_emitida']:.2f}")
            e2.metric("Absorbida", f"{metrics['e_absorbida']:.2f}")
            e3.metric("Escapada", f"{metrics['e_escapada']:.2f}")
            e4.metric("Límite de rebotes", f"{metrics['e_limite']:.2f}")

    with heat_tab:
        if mode == "2D":
            histogram, cv = calculate_heatmap_2d(st.session_state.rays)
            st.markdown('<div class="lux-section">Energía recibida a lo largo del receptor</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="lux-muted">Coeficiente de variación (CV): <b>{cv:.3f}</b> · Un CV menor indica una distribución más uniforme.</div>',
                unsafe_allow_html=True,
            )
            heat_fig = go.Figure()
            heat_fig.add_trace(go.Bar(
                x=[f"Zona {i + 1}" for i in range(len(histogram))],
                y=histogram,
                marker=dict(color="#0f9b8e", line=dict(width=0)),
                hovertemplate="%{x}<br>Energía: %{y:.2f}<extra></extra>",
            ))
            heat_fig.update_layout(
                template="plotly_white",
                height=430,
                margin=dict(l=18, r=18, t=25, b=20),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, Segoe UI, sans-serif", color="#334155"),
                xaxis_title="Posición local del receptor",
                yaxis_title="Energía acumulada",
                bargap=0.16,
            )
            heat_fig.update_xaxes(showgrid=False, linecolor="#dbe4ee")
            heat_fig.update_yaxes(gridcolor="#edf2f7", zeroline=False)
            st.plotly_chart(heat_fig, use_container_width=True, config={"displaylogo": False})
        else:
            st.info("El mapa de iluminación disponible actualmente analiza receptores 2D. La vista 3D permanece disponible en la pestaña de escena.")
else:
    st.info("Configura el ambiente en el panel lateral y pulsa **Calcular simulación** para generar las trayectorias.")
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:28px 30px;margin-top:12px;">
          <div style="font-size:1.02rem;font-weight:750;color:#17283f;margin-bottom:8px;">Cómo empezar</div>
          <div style="color:#64748b;line-height:1.8;">
            <b>1.</b> Selecciona un ambiente y una dimensión.<br>
            <b>2.</b> Ajusta la fuente, la apertura del haz y la reflectividad.<br>
            <b>3.</b> Calcula la simulación y explora las trayectorias con el control de recorrido.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
