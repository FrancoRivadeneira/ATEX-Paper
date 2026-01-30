import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parámetros
L1, L2, L3 = 0.13, 0.13, 0.13
R_torus = 0.3048
v = 0.05
z_base = 0.06 # Altura de la base del robot debido a la elevacion de las ruedas
t_total = 19.6
t_vals = np.linspace(0, t_total, 100)
arc_length = (np.pi / 2) * R_torus
D_tuberia = 0.2732
ancho_robot = 0.10
alto_robot = 0.10
tube_radius = D_tuberia / 2
margen = 0.01  # margen de seguridad para colisión (1 cm)

# Trayectoria
def posicion_orientacion(s):
    if s < 0:
        return np.array([s, R_torus]), 0
    elif s <= arc_length:
        ang = s / R_torus
        return np.array([R_torus * np.sin(ang), R_torus * np.cos(ang)]), ang + np.pi / 2
    else:
        dy = -(s - arc_length)
        return np.array([R_torus, dy]), -np.pi / 2

# Prisma 3D (cuerpo rectangular)
def prisma_3d(p_ini, p_fin, theta, beta):
    centro = (p_ini + p_fin) / 2
    largo = np.linalg.norm(p_fin - p_ini)
    ux = (p_fin - p_ini) / largo
    uy = np.array([-ux[1], ux[0]])
    verts = []
    for dz in [z_base, z_base + alto_robot]:
        base = []
        for dx, dy in [(-largo/2, -ancho_robot/2), (-largo/2, ancho_robot/2),
                       (largo/2, ancho_robot/2), (largo/2, -ancho_robot/2)]:
            offset = centro + dx * ux + dy * uy
            base.append([offset[0], offset[1], dz])
        verts.append(base)
    faces = [
        [verts[0][0], verts[0][1], verts[1][1], verts[1][0]],  # lateral
        [verts[0][1], verts[0][2], verts[1][2], verts[1][1]],  # frontal
        [verts[0][2], verts[0][3], verts[1][3], verts[1][2]],  # lateral
        [verts[0][3], verts[0][0], verts[1][0], verts[1][3]],  # trasera
        verts[0],  # base inferior
        verts[1]   # base superior
    ]
    return faces

# Distancia "signed" (clearance = tube_radius - d); >0 sin colisión, <0 colisión
# Distancia "signed" (solo se calcula en el codo)
def clearance_signed(p, R, tube_radius, arc_length):
    x, y, z = p
    if y >= 0:  # tramo curvo (codo) - solo aquí calcular clearance
        r_eje = np.sqrt(x**2 + y**2)
        d = np.sqrt((r_eje - R)**2 + (z - tube_radius)**2)
        return tube_radius - d
    else:
        # Para tramo recto no calcular clearance (retornar un valor grande positivo)
        return 1.0  # clearance positivo alto, sin riesgo



# Detección de colisión con tubería acodada (codo + tramo recto)
def colision_tuberia(p, R, tube_radius, arc_length, margen=0.01):
    # Usa el mismo criterio que clearance_signed: colisión si clearance < margen negativo
    return clearance_signed(p, R, tube_radius, arc_length) < (-margen)

# Función auxiliar para muestrear puntos dentro de un cuadrilátero 3D
def muestrear_puntos_cuadrilatero(cuad, n=5):
    """
    Dado un cuadrilátero 3D con 4 vértices [v0, v1, v2, v3],
    devuelve una malla n x n de puntos interpolados dentro del área.
    """
    v0, v1, v2, v3 = cuad
    puntos = []
    for i in range(n):
        for j in range(n):
            s = i / (n-1)
            t = j / (n-1)
            punto = (1 - s) * (1 - t) * np.array(v0) + s * (1 - t) * np.array(v1) + s * t * np.array(v2) + (1 - s) * t * np.array(v3)
            puntos.append(punto)
    return puntos

# Cálculos de trayectoria
P1, P2, P3, P4 = [], [], [], []
theta1_list, theta2_list, theta3_list = [], [], []
beta1_list, beta2_list, beta3_list = [], [], []
s_values, C_t_list = [], []

for t in t_vals:
    s = v * t
    s_values.append(s)
    p4, _ = posicion_orientacion(s)
    p3, _ = posicion_orientacion(s - L3)
    p2, _ = posicion_orientacion(s - L3 - L2)
    p1, _ = posicion_orientacion(s - L3 - L2 - L1)
    P1.append(p1)
    P2.append(p2)
    P3.append(p3)
    P4.append(p4)
    theta1_list.append(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    theta2_list.append(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]))
    theta3_list.append(np.arctan2(p4[1] - p3[1], p4[0] - p3[0]))
    beta1_list.append(0)
    beta2_list.append(0)
    beta3_list.append(0)
    # Centro local del tubo para visualización (no usado para colisión)
    if s <= arc_length:
        ang = s / R_torus
        C_t = np.array([R_torus * np.sin(ang), R_torus * np.cos(ang), tube_radius])
    else:
        C_t = np.array([R_torus, R_torus - (s - arc_length), tube_radius])
    C_t_list.append(C_t)

# Convertir listas a arrays
P1, P2, P3, P4 = map(np.array, (P1, P2, P3, P4))
theta1_list = np.array(theta1_list)
theta2_list = np.array(theta2_list)
theta3_list = np.array(theta3_list)

# ======= CÁLCULO DE CLEARANCE MÍNIMO (antes de animar) =======
def min_clearance_segment_series(P_ini_series, P_fin_series, n_sample=5):
    min_cl = np.inf
    for i in range(len(P_ini_series)):
        faces = prisma_3d(P_ini_series[i], P_fin_series[i], 0, 0)
        for face in faces:
            pts = muestrear_puntos_cuadrilatero(np.array(face), n=n_sample)
            for p in pts:
                if p[1] >= 0:  # Solo puntos en el codo
                    cl = clearance_signed(p, R_torus, tube_radius, arc_length)
                    if cl < min_cl:
                        min_cl = cl
    return min_cl


# Módulos: rear (P1-P2), middle (P2-P3), front (P3-P4)
cl_rear  = min_clearance_segment_series(P1, P2, n_sample=5)
cl_mid   = min_clearance_segment_series(P2, P3, n_sample=5)
cl_front = min_clearance_segment_series(P3, P4, n_sample=5)
cl_overall = min(cl_rear, cl_mid, cl_front)

def mm(x): return 1000.0 * x

print("\n=== Minimum Clearance Summary (3D) ===")
print(f"Rear  module : {mm(cl_rear):6.2f} mm")
print(f"Middle module: {mm(cl_mid):6.2f} mm")
print(f"Front module : {mm(cl_front):6.2f} mm")
print(f"Overall min  : {mm(cl_overall):6.2f} mm\n")

# Configuración gráfica
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.4, 0.4])
ax.set_zlim([0, 0.7])
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title("Robot tubular en 3D con detección de colisiones")

# Visualizar tubo: codo (toroide parcial)
theta = np.linspace(0, np.pi/2, 100)
phi = np.linspace(0, 2*np.pi, 40)
theta, phi = np.meshgrid(theta, phi)
x = (R_torus + tube_radius * np.cos(phi)) * np.cos(theta)
y = (R_torus + tube_radius * np.cos(phi)) * np.sin(theta)
z = tube_radius * np.sin(phi) + tube_radius
ax.plot_surface(x, y, z, color='green', alpha=0.3, linewidth=0)

# Visualizar tramo recto
rect_tube_z = np.linspace(0, tube_radius*2, 40)
rect_tube_y = np.linspace(-0.4, 0, 50)
for z_ in rect_tube_z:
    ax.plot([R_torus, R_torus], [rect_tube_y[0], rect_tube_y[-1]], [z_, z_], color='green', alpha=0.3)

# Inicializar prismas que representan segmentos del robot
robot_polys = [ax.add_collection3d(Poly3DCollection([], alpha=0.6)) for _ in range(18)]

def init():
    for poly in robot_polys:
        poly.set_verts([])
    return robot_polys

def animate(i):
    p1, p2, p3, p4 = P1[i], P2[i], P3[i], P4[i]
    count = 0
    for ini, fin, beta in zip([p1, p2, p3], [p2, p3, p4], [beta1_list[i], beta2_list[i], beta3_list[i]]):
        caras = prisma_3d(ini, fin, 0, beta)
        for cara in caras:
            cara_np = np.array(cara)
            # Muestrear puntos internos en la cara con malla 5x5
            puntos_muestra = muestrear_puntos_cuadrilatero(cara_np, n=5)
            # Verificar colisión en todos los puntos muestreados
            colisiones = [colision_tuberia(p, R_torus, tube_radius, arc_length, margen) for p in puntos_muestra]
            color = 'red' if any(colisiones) else 'blue'
            robot_polys[count].set_verts([cara_np])
            robot_polys[count].set_facecolor(color)
            count += 1
    return robot_polys

ani = FuncAnimation(fig, animate, frames=len(t_vals), init_func=init, interval=80, blit=False)
plt.tight_layout()
plt.show()
