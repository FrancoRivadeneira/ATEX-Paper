import numpy as np
import matplotlib.pyplot as plt

# Parámetros del codo y tubería
radio_codo = 304.8  # mm
diametro_tuberia = 203.2  # mm
radio_interno = radio_codo - diametro_tuberia / 2
radio_externo = radio_codo + diametro_tuberia / 2


# Definir el ángulo del codo (90° en radianes)
theta_range = np.linspace(0, np.pi / 2, 100)

# Calcular los puntos de los arcos interno y externo
x_interno = radio_interno * np.cos(theta_range)
y_interno = radio_interno * np.sin(theta_range)
x_externo = radio_externo * np.cos(theta_range)
y_externo = radio_externo * np.sin(theta_range)
x_medio = radio_codo * np.cos(theta_range)
y_medio = radio_codo * np.sin(theta_range)

# Crear figura y ejes
fig, ax = plt.subplots()

# Dibujar los bordes del codo
ax.plot(x_interno, y_interno, color='blue', label='Borde Interno')
ax.plot(x_externo, y_externo, color='red', label='Borde Externo')
ax.plot(x_medio, y_medio, color='red', label='Borde Externo')

# Dibujar las líneas que conectan los extremos
ax.plot([x_interno[0], x_interno[0]], [y_interno[0], y_externo[0]-100], color='blue')
ax.plot([x_externo[0], x_externo[0]], [y_interno[0], y_externo[0]-100], color='red')


ax.plot([x_interno[0], x_interno[0]], [y_interno[0], y_externo[0]-100], color='blue')
ax.plot([x_interno[-1], x_externo[-1]-200], [y_externo[-1], y_externo[-1]], color='red')

# Ajustar aspecto y mostrar leyenda
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.set_title("Codo de Tubería 90°")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")

plt.grid(True)
plt.show()
