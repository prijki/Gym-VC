import numpy as np

def calculate_angle_3d(a, b, c):
    # a, b, c = np.array(a), np.array(b), np.array(c)
    # ab, cb = a - b, c - b
    # cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
    # cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # return np.degrees(np.arccos(cos_angle))
    """
    Retorna o ângulo (em graus) entre os vetores (a - b) e (c - b),
    usando coordenadas 3D. Se os vetores forem degenerados retorna None.
    a, b, c: iteráveis com 2 ou 3 elementos (se tiverem z, melhor).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ab = a - b
    cb = c - b

    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)

    # evita divisão por valores próximos de zero
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return None

    cos_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_angle)))
    return angle