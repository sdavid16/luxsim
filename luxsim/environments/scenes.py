from core.surfaces import Segment2D, Circle2D, RectangularPlane3D, CylinderWall3D, Disk3D, Ellipse2D, Ellipsoid3D, PolygonPlane3D, SurfaceType

class EnvironmentLibrary:
    @staticmethod
    def get_env_A_circular(radius, ref, rec_y):
        # A: Cámara Circular 2D
        return [Circle2D([0, 0], radius, ref), Segment2D([-radius/2, rec_y], [radius/2, rec_y], 1.0, SurfaceType.RECEIVER)]
    
    @staticmethod
    def get_env_A_cylindrical(radius, h, ref, rec_z):
        # A: Cámara Cilíndrica 3D
        return [CylinderWall3D([0,0,0], radius, h, ref), Disk3D([0,0,0], [0,0,-1], radius, ref),
                Disk3D([0,0,h], [0,0,1], radius, ref), Disk3D([0,0,rec_z], [0,0,1], radius*0.8, 1.0, SurfaceType.RECEIVER)]

    @staticmethod
    def get_env_B_rectangular(w, h, ref, rec_y):
        # B: Recinto Rectangular 2D
        return [Segment2D([-w/2,-h/2], [w/2,-h/2], ref), Segment2D([w/2,-h/2], [w/2,h/2], ref),
                Segment2D([w/2,h/2], [-w/2,h/2], ref), Segment2D([-w/2,h/2], [-w/2,-h/2], ref),
                Segment2D([-w/4, rec_y], [w/4, rec_y], 1.0, SurfaceType.RECEIVER)]

    @staticmethod
    def get_env_C_elliptical(rx, ry, ref, rec_y):
        # C: Cámara Elíptica 2D
        return [Ellipse2D([0,0], rx, ry, ref), Segment2D([-rx/2, rec_y], [rx/2, rec_y], 1.0, SurfaceType.RECEIVER)]
        
    @staticmethod
    def get_env_C_ellipsoid(rx, ry, rz, ref, rec_z):
        # C: Elipsoide 3D
        return [Ellipsoid3D([0,0,0], rx, ry, rz, ref), Disk3D([0,0,rec_z], [0,0,1], rx*0.5, 1.0, SurfaceType.RECEIVER)]

    @staticmethod
    def get_env_D_gallery(w, h, ref):
        # D: Galería con Abertura Superior (Escape) 2D
        return [Segment2D([-w/2,-h/2], [w/2,-h/2], ref), Segment2D([w/2,-h/2], [w/2,h/2], ref),
                Segment2D([-w/2,h/2], [-w/2,-h/2], ref), Segment2D([-w/4, -h/2+1], [w/4, -h/2+1], 1.0, SurfaceType.RECEIVER),
                Segment2D([-w/2, h/2], [w/2, h/2], 0.0, SurfaceType.ESCAPE)] # Techo abierto

    @staticmethod
    def get_env_E_obstacle(w, h, ref):
        # E: Recinto con obstáculo circular central 2D
        surfs = EnvironmentLibrary.get_env_B_rectangular(w, h, ref, h/2 - 1)
        surfs.append(Circle2D([0, 0], w/4, 0.1)) # Obstáculo absorbente en el medio
        return surfs

    @staticmethod
    def get_env_F_guide(length, width, ref):
        # F: Conducto o Guía 2D (Fuente a la izq, Receptor a la der)
        return [Segment2D([0, -width/2], [length, -width/2], ref), Segment2D([0, width/2], [length, width/2], ref),
                Segment2D([length, -width/2], [length, width/2], 1.0, SurfaceType.RECEIVER)]

    @staticmethod
    def get_env_G_mixed_ref(w, h):
        # G: Cámara con distintas reflectividades 2D
        return [Segment2D([-w/2,-h/2], [w/2,-h/2], 0.2), Segment2D([w/2,-h/2], [w/2,h/2], 0.9), # Der alta
                Segment2D([w/2,h/2], [-w/2,h/2], 0.5), Segment2D([-w/2,h/2], [-w/2,-h/2], 0.1), # Izq baja
                Segment2D([-w/4, 0], [w/4, 0], 1.0, SurfaceType.RECEIVER)]

    @staticmethod
    def get_env_H_multi_receiver(radius, ref):
        # H: Cámara con múltiples receptores 2D
        return [Circle2D([0,0], radius, ref), 
                Segment2D([-radius/2, radius/2], [0, radius/2], 1.0, SurfaceType.RECEIVER), # Rec 1
                Segment2D([0, -radius/2], [radius/2, -radius/2], 1.0, SurfaceType.RECEIVER)] # Rec 2