import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Parámetros geométricos
R = 0.3048                      # Radio mayor del codo
D_tubo = 0.2032                 # Diámetro del tubo
r = D_tubo / 2                  # Radio del tubo (menor)
v = 0.05                        # Velocidad
arc_length = (np.pi / 2) * R
base = r

# Ángulo dentro de la sección circular del tubo para el punto deseado
Dw = 0.08
phi = - math.acos(Dw / r)  # punto en superficie inferior

# Vector fijo del punto respecto a la rueda
d = np.array([0.06, -0.01, 0.05])  # punto respecto a la rueda

# Ajuste del tiempo para iniciar desde s = -0.8 m
s_inicio = -0.8
s_fin = v * 19.6
s_vals = np.linspace(s_inicio, s_fin, 300)

# Función para la trayectoria de la rueda (sobre la superficie)
def trayectoria_rueda(s, phi):
    if s < 0:
        return np.array([s, R + r * np.cos(phi), r * np.sin(phi) + base])
    elif s <= arc_length:
        theta = s / R
        Rt = R + r * np.cos(phi)
        x = Rt * np.sin(theta)
        y = Rt * np.cos(theta)
        z = r * np.sin(phi) + base
        return np.array([x, y, z])
    else:
        delta_s = s - arc_length
        x = R + r * np.cos(phi)
        y = 0 - delta_s
        z = r * np.sin(phi) + base
        return np.array([x, y, z])

# Matriz de rotación por tramo
def matriz_rotacion(s):
    if s < 0:
        return np.eye(3)
    elif s <= arc_length:
        theta = s / R
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta),  np.cos(theta), 0],
                         [0,              0,             1]])
    else:
        return np.array([[0, 1, 0],
                         [-1,  0, 0],
                         [0,  0, 1]])

# Generar trayectorias
rueda_pts = []
punto_pts = []
for s in s_vals:
    r_w = trayectoria_rueda(s, phi)
    R_s = matriz_rotacion(s)
    p_global = r_w + R_s @ d
    rueda_pts.append(r_w)
    punto_pts.append(p_global)

rueda_pts = np.array(rueda_pts)
punto_pts = np.array(punto_pts)

# Crear el toroide (tramo de codo)
theta = np.linspace(0, np.pi/2, 100)
phi_grid = np.linspace(0, 2*np.pi, 40)
theta, phi_grid = np.meshgrid(theta, phi_grid)

x_tor = (R + r * np.cos(phi_grid)) * np.sin(theta)
y_tor = (R + r * np.cos(phi_grid)) * np.cos(theta)
z_tor = r * np.sin(phi_grid) + base

# Graficar
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Trayectorias
ax.plot(rueda_pts[:, 0], rueda_pts[:, 1], rueda_pts[:, 2],
        label="Trayectoria de la rueda", color='red', linewidth=2)
ax.plot(punto_pts[:, 0], punto_pts[:, 1], punto_pts[:, 2],
        label="Punto trasladado desde la rueda", color='orange', linewidth=2)

# Toroide transparente
ax.plot_surface(x_tor, y_tor, z_tor, color='green', alpha=0.3, linewidth=0, zorder=0)

# Configuración
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Trayectoria 3D desde s = -0.8 m")
ax.legend()
ax.grid(True)
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
