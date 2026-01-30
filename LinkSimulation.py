import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.polynomial import Polynomial

# Parámetros geométricos (del paper)
L1 = 0.35  # Distancia entre rueda trasera y primera articulación
L2 = 0.30  # Distancia entre articulaciones
L3 = 0.35  # Distancia entre segunda articulación y rueda delantera
R = 0.3048  # Radio de curvatura de la tubería
v = 0.05  # Velocidad del robot [m/s]
t_total = 19.6  # Tiempo total para atravesar la curva
t_vals = np.linspace(0, t_total, 100)  # Reducido puntos para animación más fluida
diametro_tuberia = 0.2032  # mm
radio_interno = R - diametro_tuberia / 2
radio_externo = R + diametro_tuberia / 2
theta_range = np.linspace(0, np.pi / 2, 100)

# Calcular los puntos de los arcos interno y externo
x_interno = radio_interno * np.cos(theta_range)
y_interno = radio_interno * np.sin(theta_range)
x_externo = radio_externo * np.cos(theta_range)
y_externo = radio_externo * np.sin(theta_range)

# Longitud del arco (90 grados)
arc_length = (np.pi / 2) * R

def posicion_robot(s):
    """Función de posición exactamente como la teníamos"""
    if s < 0:
        return np.array([s, R])
    elif s <= arc_length:
        theta = s / R
        x = R * np.sin(theta)
        y = R * np.cos(theta)
        return np.array([x, y])
    else:
        dy = -(s - arc_length)
        return np.array([R, dy])

# Precalculamos todas las posiciones
P1, P2, P3, P4 = [], [], [], []
for t in t_vals:
    s = v * t
    P4.append(posicion_robot(s))
    P3.append(posicion_robot(s - L3))
    P2.append(posicion_robot(s - L3 - L2))
    P1.append(posicion_robot(s - L3 - L2 - L1))

# Convertir a arrays numpy para mejor manejo
P1 = np.array(P1)
P2 = np.array(P2)
P3 = np.array(P3)
P4 = np.array(P4)

# Configuración de la figura
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(min(P1[:, 0])-0.1, max(P4[:, 0])+0.1)
ax.set_ylim(min(P4[:, 1])-0.1, max(P1[:, 1])+0.1)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel("X [m]", fontsize=12)
ax.set_ylabel("Y [m]", fontsize=12)
ax.set_title("Animación del Robot en Tubería", fontsize=14)

# Dibujar tubería (estática)
ax.plot([min(P1[:, 0])-0.1, 0], [R, R], 'k-', linewidth=3, alpha=0.3, label='Tubería')
theta_pipe = np.linspace(0, np.pi/2, 50)
x_pipe = R * np.sin(theta_pipe)
y_pipe = R * np.cos(theta_pipe)
ax.plot(x_pipe, y_pipe, 'k-', linewidth=3, alpha=0.3)
ax.plot([R, R], [0, min(P4[:, 1])-0.1], 'k-', linewidth=3, alpha=0.3)
ax.plot(x_interno, y_interno, color='blue', label='Borde Interno')
ax.plot(x_externo, y_externo, color='red', label='Borde Externo')


# Dibujar las líneas que conectan los extremos
ax.plot([x_interno[0], x_interno[0]], [y_interno[0], y_externo[0]-100], color='blue')
ax.plot([x_externo[0], x_externo[0]], [y_interno[0], y_externo[0]-100], color='red')


ax.plot([x_interno[0], x_interno[0]], [y_interno[0], y_externo[0]-100], color='blue')
ax.plot([x_interno[-1], x_externo[-1]-200], [y_externo[-1], y_externo[-1]], color='red')
# Elementos animados
line, = ax.plot([], [], 'r-', linewidth=2)  # Para el cuerpo del robot
point_p1, = ax.plot([], [], 'ro', markersize=8, label='Rueda trasera')
point_p2, = ax.plot([], [], 'go', markersize=8, label='Articulación 2')
point_p3, = ax.plot([], [], 'bo', markersize=8, label='Articulación 1')
point_p4, = ax.plot([], [], 'mo', markersize=8, label='Rueda delantera')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

# Inicialización de la animación
def init():
    line.set_data([], [])
    point_p1.set_data([], [])
    point_p2.set_data([], [])
    point_p3.set_data([], [])
    point_p4.set_data([], [])
    time_text.set_text('')
    return line, point_p1, point_p2, point_p3, point_p4, time_text

# Función de animación (CORREGIDA)
def animate(i):
    # Actualizar línea del cuerpo (conectando los puntos)
    x_vals = [P1[i, 0], P2[i, 0], P3[i, 0], P4[i, 0]]
    y_vals = [P1[i, 1], P2[i, 1], P3[i, 1], P4[i, 1]]
    line.set_data(x_vals, y_vals)
    
    # Actualizar puntos (CORRECCIÓN: usando listas de un elemento)
    point_p1.set_data([P1[i, 0]], [P1[i, 1]])
    point_p2.set_data([P2[i, 0]], [P2[i, 1]])
    point_p3.set_data([P3[i, 0]], [P3[i, 1]])
    point_p4.set_data([P4[i, 0]], [P4[i, 1]])
    
    # Actualizar texto de tiempo
    time_text.set_text(f'Tiempo: {t_vals[i]:.2f}s\n'
                     f'Posición: ({P4[i, 0]:.2f}, {P4[i, 1]:.2f})')
    
    return line, point_p1, point_p2, point_p3, point_p4, time_text

# Crear animación
ani = FuncAnimation(fig, animate, frames=len(t_vals),
                    init_func=init, blit=True, interval=50)

# Mostrar leyenda
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()