import numpy as np

def norm(v):
    """Calcula la magnitud euclidiana de un vector."""
    return np.linalg.norm(v)

def normalize(v):
    """Devuelve el vector unitario. Lanza error si la magnitud es cero."""
    n = norm(v)
    if n < 1e-12:
        raise ValueError("No se puede normalizar un vector nulo.")
    return v / n

def reflect(v_in, normal):
    """
    Calcula el vector reflejado: v_out = v_in - 2*(v_in · normal)*normal.
    Ambos vectores deben estar normalizados.
    """
    v_in_u = normalize(v_in)
    n_u = normalize(normal)
    return v_in_u - 2 * np.dot(v_in_u, n_u) * n_u