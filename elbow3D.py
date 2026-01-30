import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del toroide (codo)
R = 0.3048        # Radio del centro del toroide [m]
D = 0.2032        # Diámetro externo de la tubería [m]
r = D / 2         # Radio del tubo

# Mallas angulares
theta = np.linspace(0, np.pi/2, 100)     # solo 90° (codo)
phi = np.linspace(0, 2*np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

# Coordenadas 3D del toroide, desplazando z hacia arriba para que base esté en z=0
x = (R + r * np.cos(phi)) * np.cos(theta)
y = (R + r * np.cos(phi)) * np.sin(theta)
z = r * np.sin(phi) + r  # <<-- desplazamiento hacia arriba

# Crear figura 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Dibujar superficie
ax.plot_surface(x, y, z, color='lightgreen', alpha=0.8, edgecolor='none')

# Configuración de los ejes
ax.set_xlim([-0.6, 0.6])
ax.set_ylim([-0.6, 0.6])
ax.set_zlim([0, 0.6])
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Toroide (codo 90° con base en z=0)')

plt.tight_layout()
plt.show()
