import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------------
# Robot and pipe parameters
# -------------------------------
L1, L2, L3 = 0.13, 0.24, 0.13
R_torus = 0.381  # Torus (elbow) radius
v = 0.05
t_total = 19.6
t_vals = np.linspace(0, t_total, 100)
arc_length = (np.pi / 2) * R_torus
pipe_diameter = 0.2732
robot_width = 0.10
robot_height = 0.10
tube_radius = pipe_diameter / 2

# -------------------------------
# Trajectory and orientation
# -------------------------------
def position_orientation(s):
    if s < 0:
        return np.array([s, R_torus]), 0
    elif s <= arc_length:
        ang = s / R_torus
        return np.array([R_torus * np.sin(ang), R_torus * np.cos(ang)]), ang + np.pi / 2
    else:
        dy = -(s - arc_length)
        return np.array([R_torus, dy]), -np.pi / 2

# -------------------------------
# 3D prism builder
# -------------------------------
def prism_3d(p_ini, p_fin, theta, beta):
    center = (p_ini + p_fin) / 2
    length = np.linalg.norm(p_fin - p_ini)
    ux = (p_fin - p_ini) / length
    uy = np.array([-ux[1], ux[0]])
    verts = []

    z_base = 0.013  # Base height (13 mm)
    for dz in [z_base, z_base + robot_height]:
        base = []
        for dx, dy in [(-length/2, -robot_width/2), (-length/2, robot_width/2),
                       (length/2, robot_width/2), (length/2, -robot_width/2)]:
            offset = center + dx * ux + dy * uy
            base.append([offset[0], offset[1], dz])
        verts.append(base)

    faces = [
        [verts[0][0], verts[0][1], verts[1][1], verts[1][0]],
        [verts[0][1], verts[0][2], verts[1][2], verts[1][1]],
        [verts[0][2], verts[0][3], verts[1][3], verts[1][2]],
        [verts[0][3], verts[0][0], verts[1][0], verts[1][3]],
        verts[0], verts[1]
    ]
    return faces

# -------------------------------
# Precompute positions and angles
# -------------------------------
P1, P2, P3, P4 = [], [], [], []
theta1_list, theta2_list, theta3_list = [], [], []
beta1_list, beta2_list, beta3_list = [], [], []
s_values = []

for t in t_vals:
    s = v * t
    s_values.append(s)
    p4, _ = position_orientation(s)
    p3, _ = position_orientation(s - L3)
    p2, _ = position_orientation(s - L3 - L2)
    p1, _ = position_orientation(s - L3 - L2 - L1)

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

P1, P2, P3, P4 = map(np.array, (P1, P2, P3, P4))
theta1_list = np.array(theta1_list)
theta2_list = np.array(theta2_list)
theta3_list = np.array(theta3_list)
s_values = np.array(s_values)

# -------------------------------
# Figure setup
# -------------------------------
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.4, 0.4])
ax.set_zlim([0, 0.7])
ax.set_xlabel('X [m]', fontsize=16)
ax.set_ylabel('Y [m]', fontsize=16)
ax.set_zlabel('Z [m]', fontsize=16)
ax.set_title("3D Pipe Inspection Robot in an Elbow", fontsize=18)

# -------------------------------
# Torus surface
# -------------------------------
theta = np.linspace(0, np.pi/2, 100)
phi = np.linspace(0, 2*np.pi, 40)
theta, phi = np.meshgrid(theta, phi)

def torus_section(radius, tube_radius):
    x = (radius + tube_radius * np.cos(phi)) * np.cos(theta)
    y = (radius + tube_radius * np.cos(phi)) * np.sin(theta)
    z = tube_radius * np.sin(phi) + tube_radius
    return x, y, z

x_out, y_out, z_out = torus_section(R_torus, tube_radius)
ax.plot_surface(x_out, y_out, z_out, color='green', alpha=0.3, linewidth=0, zorder=0)

# -------------------------------
# Animation
# -------------------------------
poly_collections = []
important_frames = [0, len(t_vals)//3, 2*len(t_vals)//3, len(t_vals)-1]  # key frames

def init():
    return []

def animate(i):
    global poly_collections
    for pc in poly_collections:
        pc.remove()
    poly_collections = []

    s = s_values[i]
    bodies = [
        (P1[i], P2[i], theta1_list[i], beta1_list[i], s - L3 - L2 - L1),
        (P2[i], P3[i], theta2_list[i], beta2_list[i], s - L3 - L2),
        (P3[i], P4[i], theta3_list[i], beta3_list[i], s - L3)
    ]

    for (p_ini, p_fin, theta, beta, s_segment) in bodies:
        faces = prism_3d(p_ini, p_fin, theta, beta)
        color = 'gray' if s_segment <= arc_length else 'gray'
        poly = Poly3DCollection(faces, alpha=0.6, facecolors=color, edgecolors='k')
        ax.add_collection3d(poly)
        poly_collections.append(poly)

    # Save key frames
    if i in important_frames:
        plt.savefig(f"frame_{i}.png", dpi=300)

    return poly_collections

ani = FuncAnimation(fig, animate, frames=len(t_vals),
                    init_func=init, interval=80, blit=False)

plt.tight_layout()
plt.show()
