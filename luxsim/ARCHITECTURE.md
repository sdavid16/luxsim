# Arquitectura interna de LUXSIM

## Propósito
LUXSIM es un simulador didáctico de propagación geométrica. Las trayectorias y los impactos se calculan mediante intersecciones analíticas; Plotly solo representa el resultado. La velocidad de reproducción es visual y no representa la velocidad física de la luz.

## Flujo de ejecución
1. `app.py` recopila dimensión, escena, posición de la fuente, dirección, apertura, número de rayos, alcance y límite de rebotes.
2. `environments/scenes.py` construye las superficies analíticas de cada entorno predefinido.
3. `simulation/source.py` genera direcciones unitarias y reparte la energía total entre los rayos.
4. `simulation/engine.py` coordina el trazado y calcula métricas agregadas.
5. `core/intersections.py` busca la intersección positiva más cercana y decide si el rayo rebota, es captado, absorbido o escapa.
6. `core/ray.py` conserva la posición, energía, estado y distancia acumulada en cada evento.
7. `visualization/plotter.py` dibuja superficies y recorridos; no decide la física ni cambia las trayectorias.

## Modelo geométrico
La trayectoria rectilínea entre impactos se expresa como:

\[
\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{d},\quad t\geq 0
\]

donde \(\mathbf{p}_0\) es el origen y \(\mathbf{d}\) es una dirección unitaria. Se selecciona la primera intersección válida con \(t>\varepsilon\). En una reflexión especular ideal:

\[
\mathbf{d}_{\mathrm{salida}}=\mathbf{d}_{\mathrm{entrada}}-
2(\mathbf{d}_{\mathrm{entrada}}\cdot\mathbf{n})\mathbf{n}
\]

\(\mathbf{n}\) es la normal unitaria en el punto de impacto. La traza termina al capturarse, absorberse, escapar, alcanzar el límite de rebotes o llegar al alcance total configurado.

## Escenas y representación
- 2D: circunferencia, elipse y segmento reflectante.
- 3D: cilindro, esfera y elipsoide.
- Las superficies se dibujan con mallas o curvas; los rayos 2D son segmentos y el láser 3D es un haz tubular estrecho.
- La vista previa reactiva solo estima el primer impacto. Para obtener rebotes reales se debe pulsar **Simular**.
- El límite de alcance y el número de rebotes se pueden ajustar desde la interfaz.

## Mantenimiento
- Mantener la lógica matemática en `core/` y `simulation/`; evitar cálculos de intersección dentro de la interfaz o del plotter.
- Toda nueva superficie debe implementar `intersect(origin, direction, epsilon)` y devolver distancia positiva y normal, o `None`.
- Los cambios de visualización no deben modificar las ecuaciones, reflectividad ni estados del motor.
- Dependencias de ejecución: NumPy, Streamlit y Plotly.
