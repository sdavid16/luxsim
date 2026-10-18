import numpy as np
from .vectors import normalize

class SurfaceType:
    WALL = "PARED"
    RECEIVER = "RECEPTOR"
    ESCAPE = "ABERTURA"

class Surface:
    def __init__(self, reflectivity=1.0, surface_type=SurfaceType.WALL):
        self.reflectivity = np.clip(reflectivity, 0.0, 1.0)
        self.type = surface_type

    def intersect(self, ray_origin, ray_direction, epsilon):
        raise NotImplementedError

class Segment2D(Surface):
    def __init__(self, p1, p2, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.v = self.p2 - self.p1
        self.length = np.linalg.norm(self.v)
        self.u = self.v / self.length
        self.normal = normalize(np.array([-self.v[1], self.v[0]]))
        
    def get_uv(self, point):
        """Devuelve la distancia local (1D) a lo largo del segmento para el mapa de calor."""
        w = point - self.p1
        return np.dot(w, self.u)

    def intersect(self, ro, rd, epsilon):
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        cross_v_rd = self.v[0]*rd[1] - self.v[1]*rd[0]
        if abs(cross_v_rd) < 1e-10: 
            return None
        
        w = ro - self.p1
        t = (w[0]*self.v[1] - w[1]*self.v[0]) / cross_v_rd
        u = (w[0]*rd[1] - w[1]*rd[0]) / cross_v_rd

        if t > epsilon and 0 - 1e-8 <= u <= 1 + 1e-8:
            n = self.normal if np.dot(self.normal, rd) < 0 else -self.normal
            return t, n
        return None

class Circle2D(Surface):
    def __init__(self, center, radius, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def intersect(self, ro, rd, epsilon):
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        oc = ro - self.center
        a = np.dot(rd, rd)
        b = 2.0 * np.dot(oc, rd)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
            
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        t_hit = None
        if t1 > epsilon:
            t_hit = t1
        elif t2 > epsilon:
            t_hit = t2
            
        if t_hit is not None:
            point = ro + t_hit * rd
            normal = normalize(point - self.center)
            if np.dot(normal, rd) > 0:
                normal = -normal
            return t_hit, normal
        return None

class RectangularPlane3D(Surface):
    def __init__(self, center, u_vec, v_vec, u_size, v_size, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.center = np.array(center, dtype=float)
        self.u_dir = normalize(np.array(u_vec, dtype=float))
        self.v_dir = normalize(np.array(v_vec, dtype=float))
        self.normal = normalize(np.cross(self.u_dir, self.v_dir))
        self.u_size = u_size
        self.v_size = v_size

    def intersect(self, ro, rd, epsilon):
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        denom = np.dot(self.normal, rd)
        if abs(denom) < 1e-10:
            return None
            
        t = np.dot(self.center - ro, self.normal) / denom
        if t <= epsilon:
            return None
            
        point = ro + t * rd
        w = point - self.center
        u_proj = np.dot(w, self.u_dir)
        v_proj = np.dot(w, self.v_dir)
        
        if abs(u_proj) <= self.u_size / 2.0 and abs(v_proj) <= self.v_size / 2.0:
            n = self.normal if denom < 0 else -self.normal
            return t, n
        return None

    def get_uv(self, point):
        """Devuelve coordenadas (u, v) locales del plano."""
        w = point - self.center
        return np.dot(w, self.u_dir), np.dot(w, self.v_dir)

class Disk3D(Surface):
    """Disco circular plano en 3D (para techos, suelos o receptores circulares)."""
    def __init__(self, center, normal, radius, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.center = np.array(center, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.radius = radius

    def intersect(self, ro, rd, epsilon):
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        denom = np.dot(self.normal, rd)
        if abs(denom) < 1e-10:
            return None
            
        t = np.dot(self.center - ro, self.normal) / denom
        if t <= epsilon:
            return None
            
        point = ro + t * rd
        if np.linalg.norm(point - self.center) <= self.radius:
            n = self.normal if denom < 0 else -self.normal
            return t, n
        return None

class CylinderWall3D(Surface):
    def __init__(self, center_base, radius, height, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.cb = np.array(center_base, dtype=float)
        self.r = radius
        self.h = height

    def intersect(self, ro, rd, epsilon):
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        oc = ro[:2] - self.cb[:2]
        a = rd[0]**2 + rd[1]**2
        
        if a < 1e-10:
            return None
            
        b = 2.0 * np.dot(oc, rd[:2])
        c = np.dot(oc, oc) - self.r**2
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
            
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        for t in [t1, t2]:
            if t > epsilon:
                pz = ro[2] + t * rd[2]
                if self.cb[2] <= pz <= self.cb[2] + self.h:
                    point = ro + t * rd
                    normal = np.array([point[0] - self.cb[0], point[1] - self.cb[1], 0.0])
                    normal = normalize(normal)
                    if np.dot(normal, rd) > 0:
                        normal = -normal
                    return t, normal
        return None

class Ellipse2D(Surface):
    """Elipse en 2D centrada en los ejes."""
    def __init__(self, center, rx, ry, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.center = np.array(center, dtype=float)
        self.rx = rx
        self.ry = ry

    def intersect(self, ro, rd, epsilon):
        # BLINDAJE NUMÉRICO OBLIGATORIO:
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        oc = ro - self.center
        # Sustitución en (x/rx)^2 + (y/ry)^2 = 1
        a = (rd[0]/self.rx)**2 + (rd[1]/self.ry)**2
        if a < 1e-12: return None
        b = 2.0 * ((oc[0]*rd[0])/(self.rx**2) + (oc[1]*rd[1])/(self.ry**2))
        c = (oc[0]/self.rx)**2 + (oc[1]/self.ry)**2 - 1.0
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0: return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        t_hit = t1 if t1 > epsilon else (t2 if t2 > epsilon else None)
        if t_hit is not None:
            pt = ro + t_hit * rd
            # Normal de la elipse: vector gradiente normalizado
            grad = np.array([(pt[0]-self.center[0])/(self.rx**2), (pt[1]-self.center[1])/(self.ry**2)])
            normal = normalize(grad)
            if np.dot(normal, rd) > 0: normal = -normal
            return t_hit, normal
        return None

class Ellipsoid3D(Surface):
    """Elipsoide en 3D."""
    def __init__(self, center, rx, ry, rz, reflectivity=1.0, surface_type=SurfaceType.WALL):
        super().__init__(reflectivity, surface_type)
        self.center = np.array(center, dtype=float)
        self.rx, self.ry, self.rz = rx, ry, rz

    def intersect(self, ro, rd, epsilon):
        # BLINDAJE NUMÉRICO OBLIGATORIO:
        ro = np.array(ro, dtype=float)
        rd = np.array(rd, dtype=float)
        
        oc = ro - self.center
        a = (rd[0]/self.rx)**2 + (rd[1]/self.ry)**2 + (rd[2]/self.rz)**2
        if a < 1e-12: return None
        b = 2.0 * ((oc[0]*rd[0])/(self.rx**2) + (oc[1]*rd[1])/(self.ry**2) + (oc[2]*rd[2])/(self.rz**2))
        c = (oc[0]/self.rx)**2 + (oc[1]/self.ry)**2 + (oc[2]/self.rz)**2 - 1.0
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0: return None
        
        t1, t2 = (-b - np.sqrt(discriminant))/(2*a), (-b + np.sqrt(discriminant))/(2*a)
        t_hit = t1 if t1 > epsilon else (t2 if t2 > epsilon else None)
        if t_hit is not None:
            pt = ro + t_hit * rd
            grad = np.array([(pt[0]-self.center[0])/(self.rx**2), (pt[1]-self.center[1])/(self.ry**2), (pt[2]-self.center[2])/(self.rz**2)])
            normal = normalize(grad)
            if np.dot(normal, rd) > 0: normal = -normal
            return t_hit, normal
        return None