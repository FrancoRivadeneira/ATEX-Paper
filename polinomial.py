import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# ============================================================
# 0) CONFIG
# ============================================================
L1, L2, L3 = 0.13, 0.24, 0.13
v = 0.05
R = 0.3048
N_SAMPLES = 200

# Fig export (paper-ready)
SAVE_FIGS = True
FIG_DPI = 300
OUT_DIR = "figs_access"

# ============================================================
# 1) TRAYECTORIA 2D: STRAIGHT -> 90° ELBOW -> STRAIGHT
# ============================================================
def make_time_grid(R, v, N=200):
    arc_length = (np.pi / 2.0) * R
    t_split = arc_length / v
    t_total = 2.0 * t_split
    t_vals = np.linspace(0.0, t_total, N)
    return t_vals, t_split, t_total, arc_length

def posicion_robot(s, R, arc_length):
    if s < 0.0:
        return np.array([s, R])
    elif s <= arc_length:
        theta = s / R
        return np.array([R * np.sin(theta), R * np.cos(theta)])
    else:
        dy = -(s - arc_length)
        return np.array([R, dy])

def simulate_angles(R, v, L1, L2, L3, N=200):
    t_vals, t_split, t_total, arc_length = make_time_grid(R, v, N)
    theta1_list, theta2_list = [], []
    P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y = [], [], [], [], [], [], [], []

    for t in t_vals:
        s = v * t
        P4 = posicion_robot(s, R, arc_length)
        P3 = posicion_robot(s - L3, R, arc_length)
        P2 = posicion_robot(s - L3 - L2, R, arc_length)
        P1 = posicion_robot(s - L3 - L2 - L1, R, arc_length)

        P1_x.append(P1[0]); P1_y.append(P1[1])
        P2_x.append(P2[0]); P2_y.append(P2[1])
        P3_x.append(P3[0]); P3_y.append(P3[1])
        P4_x.append(P4[0]); P4_y.append(P4[1])

        L13 = np.linalg.norm(P1 - P3)
        L24 = np.linalg.norm(P2 - P4)

        cos_theta1 = (L1**2 + L2**2 - L13**2) / (2 * L1 * L2)
        cos_theta2 = (L2**2 + L3**2 - L24**2) / (2 * L2 * L3)
        cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

        theta1 = 180.0 - np.degrees(np.arccos(cos_theta1))
        theta2 = 180.0 - np.degrees(np.arccos(cos_theta2))
        theta1_list.append(theta1)
        theta2_list.append(theta2)

    traj = dict(
        P1_x=np.array(P1_x), P1_y=np.array(P1_y),
        P2_x=np.array(P2_x), P2_y=np.array(P2_y),
        P3_x=np.array(P3_x), P3_y=np.array(P3_y),
        P4_x=np.array(P4_x), P4_y=np.array(P4_y),
    )
    return np.array(t_vals), np.array(theta1_list), np.array(theta2_list), t_split, t_total, arc_length, traj

# ============================================================
# 2) FITS + METRICS (NO more "zero error" confusion)
# ============================================================
def fit_piecewise_quadratic(t, y, t_split):
    split_idx = np.searchsorted(t, t_split)
    t1, t2 = t[:split_idx], t[split_idx:]
    y1, y2 = y[:split_idx], y[split_idx:]
    p1 = Polynomial.fit(t1, y1, 2).convert()
    p2 = Polynomial.fit(t2, y2, 2).convert()
    yhat = np.empty_like(y)
    yhat[:split_idx] = p1(t1)
    yhat[split_idx:] = p2(t2)
    return (p1, p2, split_idx, yhat)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat)**2)))

def max_abs_err(y, yhat):
    return float(np.max(np.abs(y - yhat)))

def fit_c1_spline(t, y):
    """
    Prefer PCHIP (C¹). If SciPy missing, fallback.
    """
    try:
        from scipy.interpolate import PchipInterpolator
        spline = PchipInterpolator(t, y)
        kind = "PCHIP (C¹)"
        return spline, kind
    except Exception:
        try:
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(t, y, bc_type="natural")
            kind = "CubicSpline natural (C² fallback)"
            return spline, kind
        except Exception:
            # linear fallback
            kind = "Linear (fallback)"
            return None, kind

def eval_model(model, kind, t_eval):
    if model is None:
        # linear fallback uses interpolation on the fly must be handled outside
        raise RuntimeError("Linear fallback needs explicit handling.")
    return model(t_eval)

def continuity_jump_at_split_poly(p_left, p_right, t_split):
    """
    Jump in derivative at the split:
      |d/dt p_left(t_split) - d/dt p_right(t_split)|
    """
    dp_left = p_left.deriv()(t_split)
    dp_right = p_right.deriv()(t_split)
    return float(abs(dp_left - dp_right))

def continuity_jump_at_split_spline(spline, t_split, eps=1e-6):
    """
    Approx derivative continuity around split using finite differences:
      |theta'(t_split-ε) - theta'(t_split+ε)|
    Works for PCHIP/CubicSpline.
    """
    f = spline
    left = (f(t_split) - f(t_split - eps)) / eps
    right = (f(t_split + eps) - f(t_split)) / eps
    return float(abs(left - right))

def holdout_rmse(t, y, fit_fn, pred_fn, holdout_ratio=0.15, seed=7):
    """
    Removes a subset of points, fits on remaining, evaluates RMSE on held-out points.
    This avoids the '0 error' of exact interpolation at training points.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_hold = max(5, int(holdout_ratio * n))
    hold_idx = np.sort(idx[:n_hold])
    train_idx = np.sort(idx[n_hold:])

    t_train, y_train = t[train_idx], y[train_idx]
    t_hold, y_hold = t[hold_idx], y[hold_idx]

    model = fit_fn(t_train, y_train)
    y_pred = pred_fn(model, t_hold)
    return rmse(y_hold, y_pred)

# ============================================================
# 3) RUN (single R) + REPORT METRICS
# ============================================================
t_vals, theta1, theta2, t_split, t_total, arc_length, traj = simulate_angles(R, v, L1, L2, L3, N=N_SAMPLES)

# Piecewise quadratic
p11, p12, split_idx, th1_poly = fit_piecewise_quadratic(t_vals, theta1, t_split)
p21, p22, _,        th2_poly = fit_piecewise_quadratic(t_vals, theta2, t_split)

# Spline (C¹)
spl1, spl1_kind = fit_c1_spline(t_vals, theta1)
spl2, spl2_kind = fit_c1_spline(t_vals, theta2)

# Predictions on dense grid (for smooth plotting)
t_dense = np.linspace(t_vals[0], t_vals[-1], 800)
th1_poly_dense = np.where(
    t_dense < t_split,
    p11(t_dense),
    p12(t_dense)
)
th2_poly_dense = np.where(
    t_dense < t_split,
    p21(t_dense),
    p22(t_dense)
)

# Spline predictions
if spl1 is not None and "Linear" not in spl1_kind:
    th1_spl_dense = spl1(t_dense)
else:
    th1_spl_dense = np.interp(t_dense, t_vals, theta1)

if spl2 is not None and "Linear" not in spl2_kind:
    th2_spl_dense = spl2(t_dense)
else:
    th2_spl_dense = np.interp(t_dense, t_vals, theta2)

# Metrics to report (NO "0 error" confusion):
# 1) Error of poly on full samples (still ok)
rmse_th1_poly = rmse(theta1, th1_poly)
rmse_th2_poly = rmse(theta2, th2_poly)

# 2) Hold-out RMSE for spline and poly (fair comparison)
#    - for poly: we keep same piecewise split computed from R,v (using the same t_split)
def fit_poly_hold(t_train, y_train):
    # need t_split based on R and v, but we already have it fixed per experiment
    pL, pR, _, _ = fit_piecewise_quadratic(t_train, y_train, t_split)
    return (pL, pR)

def pred_poly_hold(model, t_eval):
    pL, pR = model
    return np.where(t_eval < t_split, pL(t_eval), pR(t_eval))

def fit_spline_hold(t_train, y_train):
    s, _ = fit_c1_spline(t_train, y_train)
    return s

def pred_spline_hold(model, t_eval):
    if model is None:
        # fallback linear
        return np.interp(t_eval, t_vals, theta1)  # won't happen if SciPy is installed
    return model(t_eval)

rmse_th1_poly_hold = holdout_rmse(t_vals, theta1, fit_poly_hold, pred_poly_hold, holdout_ratio=0.15, seed=7)
rmse_th2_poly_hold = holdout_rmse(t_vals, theta2, fit_poly_hold, pred_poly_hold, holdout_ratio=0.15, seed=7)

rmse_th1_spl_hold = holdout_rmse(t_vals, theta1, fit_spline_hold, pred_spline_hold, holdout_ratio=0.15, seed=7)
rmse_th2_spl_hold = holdout_rmse(t_vals, theta2, fit_spline_hold, pred_spline_hold, holdout_ratio=0.15, seed=7)

# 3) Continuity metric at split (key for C¹ argument)
jump_th1_poly = continuity_jump_at_split_poly(p11, p12, t_split)
jump_th2_poly = continuity_jump_at_split_poly(p21, p22, t_split)

if spl1 is not None and "Linear" not in spl1_kind:
    jump_th1_spl = continuity_jump_at_split_spline(spl1, t_split)
else:
    jump_th1_spl = np.nan

if spl2 is not None and "Linear" not in spl2_kind:
    jump_th2_spl = continuity_jump_at_split_spline(spl2, t_split)
else:
    jump_th2_spl = np.nan

print("\n=== METRICS (paper-friendly; avoids '0 error' confusion) ===")
print(f"Poly RMSE (in-sample):  θ1={rmse_th1_poly:.3f} deg, θ2={rmse_th2_poly:.3f} deg")
print(f"Holdout RMSE (15%):    θ1 poly={rmse_th1_poly_hold:.3f} deg, θ1 spline={rmse_th1_spl_hold:.3f} deg")
print(f"Holdout RMSE (15%):    θ2 poly={rmse_th2_poly_hold:.3f} deg, θ2 spline={rmse_th2_spl_hold:.3f} deg")
print(f"Derivative jump @split: θ1 poly={jump_th1_poly:.3e} deg/s, θ1 spline={jump_th1_spl:.3e} deg/s")
print(f"Derivative jump @split: θ2 poly={jump_th2_poly:.3e} deg/s, θ2 spline={jump_th2_spl:.3e} deg/s")

# ============================================================
# 4) PAPER-READY FIGURES
# ============================================================
def paper_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)

# Fig 1: Joint angles (data + fits), clean legend outside
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(t_vals, theta1, linewidth=1.8, label=r"$\theta_1$ (data)")
ax.plot(t_vals, theta2, linewidth=1.8, label=r"$\theta_2$ (data)")
ax.plot(t_dense, th1_poly_dense, linestyle="--", linewidth=1.6, label=r"$\theta_1$ (quad, piecewise)")
ax.plot(t_dense, th2_poly_dense, linestyle="--", linewidth=1.6, label=r"$\theta_2$ (quad, piecewise)")
ax.plot(t_dense, th1_spl_dense, linestyle=":", linewidth=2.0, label=rf"$\theta_1$ ({spl1_kind})")
ax.plot(t_dense, th2_spl_dense, linestyle=":", linewidth=2.0, label=rf"$\theta_2$ ({spl2_kind})")
ax.axvline(t_split, color="gray", linestyle="--", linewidth=1.2)
ax.set_xlabel("Time [s]", fontsize=11)
ax.set_ylabel("Angle [deg]", fontsize=11)
ax.set_title(rf"Joint angle reconstruction ($R={R:.3f}$ m, $v={v:.2f}$ m/s)", fontsize=12)
paper_axes(ax)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=9)

fig.tight_layout()
if SAVE_FIGS:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/angles_fits_R{R:.3f}.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# Fig 2: Error around split (zoomed region), more informative than absolute error everywhere
zoom_w = 0.15 * t_total  # 15% window around split
t_min = max(t_vals[0], t_split - zoom_w)
t_max = min(t_vals[-1], t_split + zoom_w)

mask_dense = (t_dense >= t_min) & (t_dense <= t_max)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
# Plot absolute error for poly and spline in zoom window
# For spline, evaluate error on dense grid against linear interp "data curve" for visualization:
theta1_dense_data = np.interp(t_dense, t_vals, theta1)
theta2_dense_data = np.interp(t_dense, t_vals, theta2)

ax.plot(t_dense[mask_dense], np.abs(theta1_dense_data[mask_dense] - th1_poly_dense[mask_dense]),
        linewidth=1.8, label=r"$|\theta_1 - \hat{\theta}_1|$ (poly)")
ax.plot(t_dense[mask_dense], np.abs(theta1_dense_data[mask_dense] - th1_spl_dense[mask_dense]),
        linewidth=1.8, linestyle=":", label=r"$|\theta_1 - \hat{\theta}_1|$ (spline)")

ax.axvline(t_split, color="gray", linestyle="--", linewidth=1.2)
ax.set_xlabel("Time [s]", fontsize=11)
ax.set_ylabel("Abs error [deg]", fontsize=11)
ax.set_title("Zoomed reconstruction error near elbow exit (around $t_{split}$)", fontsize=12)
paper_axes(ax)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig(f"{OUT_DIR}/error_zoom_theta1_R{R:.3f}.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# ============================================================
# 5) MAX ANGLE vs R (paper-ready)
# ============================================================
R_list = np.linspace(0.15, 0.60, 12)
max_th1, max_th2, max_th = [], [], []

for Rk in R_list:
    tk, th1k, th2k, t_splitk, t_totk, arc_k, _ = simulate_angles(Rk, v, L1, L2, L3, N=N_SAMPLES)
    max1 = float(np.max(th1k))
    max2 = float(np.max(th2k))
    max_th1.append(max1)
    max_th2.append(max2)
    max_th.append(max(max1, max2))

max_th1 = np.array(max_th1)
max_th2 = np.array(max_th2)
max_th = np.array(max_th)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(R_list, max_th1, marker="o", linewidth=2.0, label=r"$\max(\theta_1)$")
ax.plot(R_list, max_th2, marker="o", linewidth=2.0, label=r"$\max(\theta_2)$")
ax.plot(R_list, max_th,  marker="s", linewidth=2.2, label=r"$\max(\max(\theta_1,\theta_2))$")
ax.set_xlabel("Elbow radius $R$ [m]", fontsize=11)
ax.set_ylabel("Max joint angle [deg]", fontsize=11)
ax.set_title(rf"Max joint angles vs elbow radius ($v={v:.2f}$ m/s)", fontsize=12)
paper_axes(ax)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
if SAVE_FIGS:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/max_angle_vs_R.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

# ============================================================
# 6) Trajectory (cleaner)
# ============================================================
fig, ax = plt.subplots(figsize=(6.4, 5.2))
# Pipe
ax.plot([np.min(traj["P1_x"])-0.1, 0.0], [R, R], 'k-', linewidth=2.5, alpha=0.25)
theta_pipe = np.linspace(0.0, np.pi/2.0, 80)
ax.plot(R*np.sin(theta_pipe), R*np.cos(theta_pipe), 'k-', linewidth=2.5, alpha=0.25)
ax.plot([R, R], [0.0, np.min(traj["P4_y"])-0.1], 'k-', linewidth=2.5, alpha=0.25)

# Robot points
ax.plot(traj["P1_x"], traj["P1_y"], linewidth=2.0, label="P1 (rear wheel)")
ax.plot(traj["P2_x"], traj["P2_y"], linewidth=2.0, label="P2 (rear joint)")
ax.plot(traj["P3_x"], traj["P3_y"], linewidth=2.0, label="P3 (front joint)")
ax.plot(traj["P4_x"], traj["P4_y"], linewidth=2.0, label="P4 (front wheel)")

ax.set_xlabel("X [m]", fontsize=11)
ax.set_ylabel("Y [m]", fontsize=11)
ax.set_title(rf"Centerline trajectory and key points ($R={R:.3f}$ m)", fontsize=12)
ax.axis("equal")
paper_axes(ax)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
fig.tight_layout()
if SAVE_FIGS:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/trajectory_R{R:.3f}.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()
