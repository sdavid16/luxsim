import streamlit as st
import numpy as np
from simulation.source import Source2D, Source3D
from environments.scenes import EnvironmentLibrary
from simulation.engine import run_simulation
from visualization.plotter import plot_scene
from simulation.analysis import calculate_heatmap_2d
import plotly.graph_objects as go

st.set_page_config(page_title="LUXSIM Avanzado", layout="wide")
st.title("LUXSIM - Simulación Analítica y Animación")

# Estados
if "rays" not in st.session_state:
    st.session_state.update({'rays': [], 'metrics': {}, 'surfaces': [], 'max_sim_dist': 100.0, 'calc_done': False})

tabs = st.tabs(["1. Entorno y Fuente", "2. Simulación y Animación", "3. Mapa de Iluminación", "4. Optimización"])

with tabs[0]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Biblioteca de Ambientes")
        amb_sel = st.selectbox("Escenario", ["A. Cámara Circular/Cilíndrica", "B. Recinto Rectangular", "C. Cámara Elíptica", "D. Galería con Abertura", "E. Obstáculo Interior", "F. Guía Luminosa", "G. Reflectividad Mixta", "H. Múltiples Receptores"])
        modo = st.radio("Dimensionalidad", ["2D", "3D"])
        
    with col2:
        st.subheader("Propiedades Físicas")
        ref_wall = st.slider("Reflectividad Media (ρ)", 0.0, 1.0, 0.8)
        max_b = st.number_input("Límite de Rebotes", 1, 100, 20)
        
    with col3:
        st.subheader("Fuente Luminosa")
        n_rays = st.slider("Número de Rayos", 1, 200, 25)
        ang_c = st.slider("Ángulo de Emisión", 0, 360, 90)
        ang_s = st.slider("Amplitud del Haz", 1, 360, 60)

    if st.button("Calcular Trazado Analítico", type="primary"):
        # Construcción dinámica (Selección de entorno)
        if "A." in amb_sel:
            surfs = EnvironmentLibrary.get_env_A_circular(5, ref_wall, 3) if modo=="2D" else EnvironmentLibrary.get_env_A_cylindrical(5, 10, ref_wall, 8)
        elif "B." in amb_sel:
            surfs = EnvironmentLibrary.get_env_B_rectangular(10, 10, ref_wall, 4) if modo=="2D" else [] # ... Omitido por brevedad
        elif "C." in amb_sel:
            surfs = EnvironmentLibrary.get_env_C_elliptical(6, 4, ref_wall, 2) if modo=="2D" else EnvironmentLibrary.get_env_C_ellipsoid(6, 4, 8, ref_wall, 4)
        else:
            surfs = EnvironmentLibrary.get_env_D_gallery(10, 10, ref_wall) # Fallback didáctico
            
        src = Source2D([0, -2], ang_c, ang_s, n_rays, 1000.0) if modo=="2D" else Source3D([0,0,1], [0,0,1], ang_s, n_rays, 1000.0)
        
        rays, metrics = run_simulation(src, surfs, max_b)
        
        # Guardar en sesión para aislar cálculo de animación
        max_d = max([r.cumulative_distance for r in rays]) if rays else 100.0
        st.session_state.update({'rays': rays, 'metrics': metrics, 'surfaces': surfs, 'max_sim_dist': max_d, 'calc_done': True, 'modo': modo})
        st.success("Cálculo termodinámico completado.")

with tabs[1]:
    if st.session_state.calc_done:
        st.subheader("Animación Temporal Interactiva")
        st.info("La velocidad de la línea de tiempo recorta los historiales calculados analíticamente, dibujando progresivamente.")
        
        col_anim, col_met = st.columns([3, 1])
        with col_anim:
            # Control de animación temporal
            time_slider = st.slider("Progreso de Animación (Distancia Recorrida)", 0.0, st.session_state.max_sim_dist, st.session_state.max_sim_dist, 0.5)
            
            # Renderizado progresivo
            fig = plot_scene(st.session_state.surfaces, st.session_state.rays, st.session_state.modo=="3D", time_slider)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_met:
            m = st.session_state.metrics
            st.metric("Energía Captada", f"{m['e_captada']:.1f}", f"{m['eficiencia_pct']:.1f}% Efi")
            st.metric("Absorbida / Escapada", f"{m['e_absorbida']:.1f} / {m['e_escapada']:.1f}")
            st.metric("Rebotes Promedio", f"{m['rebotes_medios']:.1f}")

with tabs[2]:
    if st.session_state.calc_done and st.session_state.modo == "2D":
        hist, cv = calculate_heatmap_2d(st.session_state.rays)
        st.subheader(f"Uniformidad de Iluminación (CV: {cv:.3f})")
        fig_heat = go.Figure(data=[go.Bar(y=hist, marker_color='green')])
        fig_heat.update_layout(title="Distribución de Energía en el Receptor", xaxis_title="Posición Local (UV)", yaxis_title="Energía Acumulada")
        st.plotly_chart(fig_heat, use_container_width=True)