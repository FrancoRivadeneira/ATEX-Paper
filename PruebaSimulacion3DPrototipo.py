import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parámetros geométricos
L1, L2, L3 = 0.2, 0.30, 0.2
R = 0.3048
r = 0.2032 / 2
v = 0.05
H = 0.1  # altura del robot
W = 0.1  # ancho del robot
phi_contact = -np.pi / 2  # contacto inferior
d = np.array([0.06, -0.01, 0.05]) 
t_total = 19.6
t_vals = np.linspace(0, t_total, 100)
s_vals = v * t_vals
arc_length = (np.pi / 2) * R

# Trayectoria en 3D
def trayectoria(s, phi):
    if s < 0:
        return np.array([s, R + r * np.cos(phi), r * np.sin(phi)])
    elif s <= arc_length:
        theta = s / R
        Rt = R + r * np.cos(phi)
        return np.array([Rt * np.sin(theta), Rt * np.cos(theta), r * np.sin(phi)])
    else:
        delta = s - arc_length
        return np.array([R + r * np.cos(phi), -delta, r * np.sin(phi)])

# Matriz de rotación local
def rotacion(s):
    if s < 0:
        return np.eye(3)
    elif s <= arc_length:
        theta = s / R
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    else:
        return np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]])
def trayectoriaP4(s,phi):
    return trayectoria(s, phi)+rotacion(s)@d

# Cálculo de posiciones y orientaciones
def cuerpo_vertices(p_ini, p_fin, ancho=W, alto=H):
    eje_x = (p_fin - p_ini)
    largo = np.linalg.norm(eje_x)
    eje_x /= largo
    eje_z = np.array([0, 0, 1])
    eje_y = np.cross(eje_z, eje_x)
    eje_y /= np.linalg.norm(eje_y)

    centro = (p_ini + p_fin) / 2
    dx = (largo / 2) * eje_x
    dy = (ancho / 2) * eje_y
    dz = (alto / 2) * eje_z

    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corner = centro + sx * dx + sy * dy + sz * dz
                corners.append(corner)
    faces = [
        [corners[i] for i in [0,1,3,2]],
        [corners[i] for i in [4,5,7,6]],
        [corners[i] for i in [0,1,5,4]],
        [corners[i] for i in [2,3,7,6]],
        [corners[i] for i in [0,2,6,4]],
        [corners[i] for i in [1,3,7,5]],
    ]
    return faces

# Posiciones
P1, P2, P3, P4 = [], [], [], []
for s in s_vals:
    P4.append(trayectoriaP4(s, phi_contact))
    P3.append(trayectoriaP4(s - L3, phi_contact))
    P2.append(trayectoriaP4(s - L3 - L2, phi_contact))
    P1.append(trayectoriaP4(s - L3 - L2 - L1, phi_contact))

# Crear figura
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 0.4])
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Simulación 3D del Robot Multicuerpo")

polycollections = []

def init():
    return []

def update(i):
    global polycollections
    for p in polycollections:
        p.remove()
    polycollections.clear()

    cuerpos = [
        (P1[i], P2[i]),
        (P2[i], P3[i]),
        (P3[i], P4[i])
    ]

    for ini, fin in cuerpos:
        faces = cuerpo_vertices(np.array(ini), np.array(fin))
        poly = Poly3DCollection(faces, alpha=0.6, facecolors='gray', edgecolor='black')
        ax.add_collection3d(poly)
        polycollections.append(poly)

    return polycollections

ani = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, blit=False)
plt.tight_layout()
plt.show()
