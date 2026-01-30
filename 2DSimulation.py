import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Robot and pipe parameters
L1, L2, L3 = 0.2, 0.30, 0.2
R = 0.3048
v = 0.05
t_total = 19.6
t_vals = np.linspace(0, t_total, 200)
arc_length = (np.pi / 2) * R
pipe_diameter = 0.2032
robot_width = 0.10  # 10 cm

# Pipe edges
inner_radius = R - pipe_diameter / 2
outer_radius = R + pipe_diameter / 2
theta_range = np.linspace(0, np.pi / 2, 100)
x_inner = inner_radius * np.cos(theta_range)
y_inner = inner_radius * np.sin(theta_range)
x_outer = outer_radius * np.cos(theta_range)
y_outer = outer_radius * np.sin(theta_range)

# Path and orientation function
def position_orientation(s):
    if s < 0:
        return np.array([s, R]), 0
    elif s <= arc_length:
        ang = s / R
        return np.array([R * np.sin(ang), R * np.cos(ang)]), ang + np.pi / 2
    else:
        dy = -(s - arc_length)
        return np.array([R, dy]), -np.pi / 2

# Precompute positions and orientations
P1, P2, P3, P4 = [], [], [], []
theta1_list, theta2_list, theta3_list = [], [], []

for t in t_vals:
    s = v * t
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

P1, P2, P3, P4 = map(np.array, (P1, P2, P3, P4))
theta1_list, theta2_list, theta3_list = map(np.array, (theta1_list, theta2_list, theta3_list))

# Figure setup
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(min(P1[:, 0]) - 0.1, max(P4[:, 0]) + 0.3)
ax.set_ylim(min(P4[:, 1]) - 0.1, max(P1[:, 1]) + 0.3)
ax.set_aspect('equal')
ax.grid(True)
# ax.set_title("Robot with Rectangular Bodies", fontsize=18)
ax.set_xlabel("X [m]", fontsize=25)
ax.set_ylabel("Y [m]", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=14)

# Pipe
ax.plot(x_inner, y_inner, 'b--', alpha=0.5, label="Inner edge")
ax.plot(x_outer, y_outer, 'r--', alpha=0.5, label="Outer edge")

# Animated elements
rects = [plt.Rectangle((0, 0), 0, 0, angle=0, color='gray', alpha=0.6) for _ in range(3)]
for rect in rects:
    ax.add_patch(rect)
line_body, = ax.plot([], [], 'k-', linewidth=2)
points, = ax.plot([], [], 'ko', markersize=6)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=25)

# Init function
def init():
    line_body.set_data([], [])
    points.set_data([], [])
    for rect in rects:
        rect.set_xy((0, 0))
        rect.set_angle(0)
    return rects + [line_body, points, time_text]

# Animation update
def animate(i):
    x = [P1[i, 0], P2[i, 0], P3[i, 0], P4[i, 0]]
    y = [P1[i, 1], P2[i, 1], P3[i, 1], P4[i, 1]]
    line_body.set_data(x, y)
    points.set_data(x, y)

    bodies = [(P3[i], P4[i], theta3_list[i]),
              (P2[i], P3[i], theta2_list[i]),
              (P1[i], P2[i], theta1_list[i])]

    for j, (p_ini, p_fin, theta) in enumerate(bodies):
        center = (p_ini + p_fin) / 2
        length = np.linalg.norm(p_fin - p_ini)
        dx = robot_width * np.sin(theta) / 2
        dy = -robot_width * np.cos(theta) / 2
        lower_corner = center + np.array([-length/2 * np.cos(theta) + dx,
                                          -length/2 * np.sin(theta) + dy])
        rects[j].set_xy(lower_corner)
        rects[j].set_width(length)
        rects[j].set_height(robot_width)
        rects[j].angle = np.degrees(theta)

    time_text.set_text(f"Time: {t_vals[i]:.2f} s")
    return rects + [line_body, points, time_text]

# Save snapshots of key instants
key_times = [0, arc_length / v, (arc_length + 0.1) / v, t_total]  # [start, start curve, end curve, end]
for kt in key_times:
    idx = (np.abs(t_vals - kt)).argmin()
    animate(idx)
    plt.savefig(f"robot_snapshot_{kt:.2f}s.png", dpi=300)

# Animation
ani = FuncAnimation(fig, animate, init_func=init, frames=len(t_vals),
                    interval=50, blit=False)

plt.legend(fontsize=20)
plt.tight_layout()
plt.show()
