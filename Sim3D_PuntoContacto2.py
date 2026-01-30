import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Parámetros geométricos
R = 0.3048                      # Radio mayor del codo
D_tubo = 0.2032                 # Diámetro del tubo
r = D_tubo / 2                  # Radio del tubo (menor)
v = 0.05                        # Velocidad
t_total = 19.6
t_vals = np.linspace(0, t_total, 300)
arc_length = (np.pi / 2) * R
base = r

# Ángulo dentro de la sección circular del tubo para el punto deseado
Dw = 0.08                        # Nueva variable
phi = np.pi+ math.acos(Dw / r)         # phi calculado según Dw y r

# Función para la trayectoria del centro
def trayectoria_centro(s):
    if s < 0:
        return np.array([s, R, base])
    elif s <= arc_length:
        theta = s / R
        return np.array([R * np.sin(theta), R * np.cos(theta), base])
    else:
        dy = -(s - arc_length)
        return np.array([R, dy, base])

# Función para la trayectoria del punto en la superficie del toroide
def trayectoria_punto(s, phi):
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

# Generar trayectorias
centro_pts = []
punto_pts = []
for t in t_vals:
    s = v * t
    centro_pts.append(trayectoria_centro(s))
    punto_pts.append(trayectoria_punto(s, phi))

centro_pts = np.array(centro_pts)
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
ax.plot(centro_pts[:, 0], centro_pts[:, 1], centro_pts[:, 2],
        label="Centro del tubo", color='blue', linewidth=2)
ax.plot(punto_pts[:, 0], punto_pts[:, 1], punto_pts[:, 2],
        label=f"Punto sobre la pared\n(φ = {phi:.2f} rad)", color='red', linewidth=2)

# Toroide transparente
ax.plot_surface(x_tor, y_tor, z_tor, color='green', alpha=0.3, linewidth=0, zorder=0)

# Configuración
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Trayectoria 3D sobre codo: Centro vs Superficie del tubo")
ax.legend()
ax.grid(True)
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
