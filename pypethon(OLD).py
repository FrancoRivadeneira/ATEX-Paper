import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
R = 304.8  # Radio del codo (mm)
L = 350    # Distancia entre los ejes delanteros (mm)
D = 304.23  # Distancia entre las ruedas traseras (mm)

class DifferentialRobot:
    def __init__(self, wheel_distance, wheel_radius):
        self.wheel_distance = wheel_distance
        self.wheel_radius = wheel_radius
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def kinematic(self, theta, beta, alpha):
        Rot = np.array([[np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        
        W_frontal = np.array([[L, L, L],
                              [0.5 * D * np.cos(beta), -0.5 * D * np.cos(beta), 0],
                              [-0.5 * D * np.sin(beta), -0.5 * D * np.sin(beta), 0]])
        W_rear = np.array([[0, 0, 0],
                           [0.5 * D * np.cos(beta), -0.5 * D * np.cos(beta),0],
                           [-0.5 * D * np.sin(beta), -0.5 * D * np.sin(beta),0]])
        
        term = np.sqrt(max(0, L**2 - R**2 * (1 - np.cos(alpha))**2))
        To = np.array([[R * np.sin(alpha) - term, R * np.sin(alpha) - term,R * np.sin(alpha) - term],
                       [R, R, R],
                       [0, 0, 0]])
        
        pos = Rot @ W_frontal + To 
        pos_rear = Rot @ W_rear + To 
        
        x_izq_del = pos[0][0]
        y_izq_del = pos[1][0]
        x_der_del = pos[0][1]
        y_der_del = pos[1][1]
        x_cen_del = pos[0][2]
        y_cen_del = pos[1][2]
        x_izq_tra = pos_rear[0][0]
        y_izq_tra = pos_rear[1][0]
        x_der_tra = pos_rear[0][1]
        y_der_tra = pos_rear[1][1]
        x_cen_tra = pos_rear[0][2]
        y_cen_tra = pos_rear[1][2]

        return x_izq_del, y_izq_del, x_der_del, y_der_del, x_izq_tra, y_izq_tra, x_der_tra, y_der_tra,x_cen_del,y_cen_del,x_cen_tra,y_cen_tra
   
    
    def update_position(self, v_left, v_right, dt):
        v_l = v_left * self.wheel_radius
        v_r = v_right * self.wheel_radius
        v = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.wheel_distance
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        
    def get_position(self):
        return self.x, self.y, self.theta


# Parámetros del robot
wheel_distance = 122  # Distancia entre las ruedas (mm)
wheel_radius = 50  # Radio de las ruedas (mm)

# Crear el primer robot
robot1 = DifferentialRobot(wheel_distance, wheel_radius)

# Crear el segundo robot
robot2 = DifferentialRobot(wheel_distance, wheel_radius)

# Crear el tercer robot
robot3 = DifferentialRobot(wheel_distance, wheel_radius)


# Parámetros del codo y tubería
radio_codo = 304.8  # mm
diametro_tuberia = 203.2  # mm
radio_interno = radio_codo - diametro_tuberia / 2
radio_externo = radio_codo + diametro_tuberia / 2

# Definir el ángulo del codo (90° en radianes)
theta_range = np.linspace(0, np.pi/2, 100)

# Calcular los puntos de los arcos interno y externo
x_interno = radio_interno * np.cos(theta_range)
y_interno = radio_interno * np.sin(theta_range) 
x_externo = radio_externo * np.cos(theta_range)
y_externo = radio_externo * np.sin(theta_range) 

# Configurar aspecto de la gráfica
fig, ax = plt.subplots()
ax.plot(x_interno, y_interno, color='blue', label='Borde Interno')
ax.plot(x_externo, y_externo, color='red', label='Borde Externo')
ax.plot([x_interno[0], x_externo[0]], [y_interno[0], y_externo[0]], color='black')
ax.plot([x_interno[-1], x_externo[-1]], [y_interno[-1], y_externo[-1]], color='black')
ax.set_aspect('equal', adjustable='box')

# Función de animación
# Función de animación
flag=False
x2_izq_del=0
y2_izq_del=0
x2_der_del=0
y2_der_del=0
x2_izq_tra=0
y2_izq_tra=0
x2_der_tra=0
y2_der_tra=0
x2_cen_del=0
y2_cen_del=0
x2_cen_tra=0
y2_cen_tra=0
ancho=0
theta_ant=80
def animate(i):
    global flag,x2_izq_del,y2_izq_del,x2_der_del,y2_der_del,x2_izq_tra,y2_izq_tra,x2_der_tra,y2_der_tra,x2_cen_del,y2_cen_del,ancho,theta_ant,x2_cen_tra,y2_cen_tra
    ax.clear()
    ax.plot(x_interno, y_interno, color='blue', label='Borde Interno')
    ax.plot(x_externo, y_externo, color='red', label='Borde Externo')
    ax.plot([x_interno[0], x_externo[0]], [y_interno[0], y_interno[0]], color='black')
    ax.plot([x_interno[-1], x_externo[-1]], [y_interno[-1], y_externo[-1]], color='black')
    ax.set_aspect('equal', adjustable='box')

    alpha = np.radians(i)
    beta = np.radians(65)
    theta1 = 0.33357872 + 0.21694819 * np.degrees(alpha) - 0.00520002 * (np.degrees(alpha))**2 + 0.00010575 * (np.degrees(alpha))**3
    theta2 = theta1  # El segundo robot tiene la misma orientación que el primero
    
    # Posiciones del primer robot
    x1_izq_del, y1_izq_del, x1_der_del, y1_der_del, x1_izq_tra, y1_izq_tra, x1_der_tra, y1_der_tra,x1_cen_del,y1_cen_del,x1_cen_tra,y1_cen_tra = robot1.kinematic(np.radians(theta1), beta, alpha)
    # x1_cen_tra = (x1_der_tra+x1_izq_tra)/2
    # y1_cen_tra = (y1_der_tra+y1_izq_tra)/2
    # Dibujar el primer robot
    ax.plot([x1_izq_del, x1_der_del], [y1_izq_del, y1_der_del], color='purple', linestyle='--', label='Línea Frontal Robot 1')
    ax.scatter([x1_izq_del, x1_der_del], [y1_izq_del, y1_der_del], color='red')
    ax.plot([x1_izq_tra, x1_der_tra], [y1_izq_tra, y1_der_tra], color='purple', linestyle='--', label='Línea Trasera Robot 1')
    ax.scatter([x1_izq_tra, x1_der_tra], [y1_izq_tra, y1_der_tra], color='red')
    ax.plot([x1_izq_tra, x1_izq_del], [y1_izq_tra, y1_izq_del], color='purple', linestyle='--', label='Línea Lateral Izquierda Robot 1')
    ax.plot([x1_der_tra, x1_der_del], [y1_der_tra, y1_der_del], color='purple', linestyle='--', label='Línea Lateral Derecha Robot 1')
    ax.scatter(x1_cen_tra, y1_cen_tra, color='red')
    ax.scatter(x1_cen_del, y1_cen_del, color='red')
    # Verificar si theta1 es mayor a 45 grados
    if not flag:
        x2_izq_del = x1_izq_tra
        y2_izq_del = y1_izq_tra
        x2_der_del = x1_der_tra
        y2_der_del = y1_der_tra

        x2_cen_del = (x1_der_tra+x1_izq_tra)/2
        y2_cen_del = (y1_der_tra+y1_izq_tra)/2

        ancho=y1_izq_tra-y1_der_tra

        x2_izq_tra = x2_izq_del - L
        y2_izq_tra = y2_izq_del 
        x2_der_tra = x2_der_del - L
        y2_der_tra = y2_der_del 
        x2_cen_tra = (x2_der_tra+x2_izq_tra)/2
        y2_cen_tra = (y2_der_tra+y2_izq_tra)/2
        flag=True
        x2_izq_del = x1_cen_tra
        y2_izq_del = y1_cen_tra+ancho/2
        x2_der_del = x1_cen_tra
        y2_der_del = y1_cen_tra-ancho/2

        x2_izq_tra = x2_izq_del - L
        y2_izq_tra = y2_izq_del 
        x2_der_tra = x2_der_del - L
        y2_der_tra = y2_der_del 
        # Dibujar el segundo robot
        ax.plot([x2_izq_del, x2_der_del], [y2_izq_del, y2_der_del], color='green', linestyle='--', label='Línea Frontal Robot 2')
        ax.scatter([x2_izq_del, x2_der_del], [y2_izq_del, y2_der_del], color='blue')
        ax.plot([x2_izq_tra, x2_der_tra], [y2_izq_tra, y2_der_tra], color='green', linestyle='--', label='Línea Trasera Robot 2')
        ax.scatter([x2_izq_tra, x2_der_tra], [y2_izq_tra, y2_der_tra], color='blue')
        ax.plot([x2_izq_tra, x2_izq_del], [y2_izq_tra, y2_izq_del], color='green', linestyle='--', label='Línea Lateral Izquierda Robot 2')
        ax.plot([x2_der_tra, x2_der_del], [y2_der_tra, y2_der_del], color='green', linestyle='--', label='Línea Lateral Derecha Robot 2')
        ax.scatter(x2_cen_del, y2_cen_del, color='red')

    
# Crear la animación
ani = FuncAnimation(fig, animate, frames=100, interval=100)

# Mostrar la animación
plt.show()
