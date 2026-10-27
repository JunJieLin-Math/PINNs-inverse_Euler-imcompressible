
"""
2D PINN inverse problem:
Recover M, R, d, T from one synthetic X-ray-like image.

Physics:
    u(y) = -U0 * sin(2*pi*y/Ly)
    v(y) = 0
    c_t + u(y) * c_x = 0

Normalized time:
    tau = t / T
    c_tau + T*u(y)*c_x = 0

Initial condition:
    c(x,y,0) = g1(x,y) + M*g2(x,y)

Synthetic observation:
    I(x,y) = c(x,y,T)^2

Important:
- The velocity field is prescribed, not learned.
- This is a passive-scalar inverse-problem PoC,
  NOT a full Euler simulation or a physical cluster model.
- M is a scalar-amplitude ratio in this PoC, not automatically
  the true total-mass ratio of galaxy clusters.
- The X-ray model I ~ c^2 assumes a simplified fixed-temperature
  emissivity proxy.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Configuration
# ============================================================

# Periodic spatial domain
LX = 10.0
LY = 8.0

# Prescribed Euler-compatible shear flow
U0 = 1.0

# Fixed initial horizontal half-separation
# This MUST remain fixed in the first PoC, otherwise it can
# be confounded with the unknown observation time T.
A = 1.0

# Reference Gaussian width
R1 = 0.45

# Ground-truth parameters used ONLY to generate synthetic data.
# The inverse model does not receive these as initial guesses.
TRUE_PARAMS = {
    "M": 1.35,
    "R": 1.25,
    "d": 1.25,
    "T": 5.00,
}

# Parameter bounds used by the inverse model
PARAM_BOUNDS = {
    "M": (0.40, 2.00),
    "R": (0.65, 1.60),
    "d": (0.60, 2.00),
    "T": (0.20, 7.50),
}

# Synthetic image resolution
NX = 96
NY = 96


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_sigmoid(z):
    """Inverse of sigmoid, with numerical clipping."""
    z = min(max(float(z), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(z / (1.0 - z))


def sample_uniform(n, device):
    """Uniform samples in the physical spatial domain."""
    x = (torch.rand(n, 1, device=device) - 0.5) * LX
    y = (torch.rand(n, 1, device=device) - 0.5) * LY
    return x, y


def shear_velocity(y):
    """
    Prescribed incompressible Euler-compatible shear flow:
        u(y) = -U0 sin(2*pi*y/LY)
        v = 0
    """
    u = -U0 * torch.sin(2.0 * math.pi * y / LY)
    return u


# ============================================================
# 3. Periodic Gaussian initial condition
# ============================================================

def periodic_gaussian(x, y, cx, cy, sigma):
    """
    Periodized 2D Gaussian on the rectangular torus.

    A finite image sum approximates:
        g(x,y) = sum_{k,l in Z} Gaussian(x+k*LX, y+l*LY)

    The prefactor is the usual 2D Gaussian normalization.
    For the chosen domain and widths, the omitted image terms
    are negligible.

    x, y: tensors with compatible shapes
    cx, cy, sigma: scalar tensors or Python floats
    """
    px = torch.zeros_like(x)
    py = torch.zeros_like(y)

    # A few periodic copies are sufficient for this domain.
    for k in (-1, 0, 1):
        dx = x - cx + k * LX
        px = px + torch.exp(-0.5 * (dx / sigma) ** 2)

    for k in (-1, 0, 1):
        dy = y - cy + k * LY
        py = py + torch.exp(-0.5 * (dy / sigma) ** 2)

    return (px * py) / (2.0 * math.pi * sigma**2)


def initial_scalar(x, y, M, R, d):
    """
    c0 = g1 + M*g2

    Centers:
        cluster 1: (-A, -d/2)
        cluster 2: (+A, +d/2)

    Widths:
        sigma1 = R1
        sigma2 = R * R1

    Integrated scalar strengths are approximately 1 and M.
    """
    sigma1 = torch.as_tensor(R1, device=x.device, dtype=x.dtype)
    sigma2 = R * R1

    g1 = periodic_gaussian(
        x=x,
        y=y,
        cx=-A,
        cy=-0.5 * d,
        sigma=sigma1,
    )

    g2 = periodic_gaussian(
        x=x,
        y=y,
        cx=+A,
        cy=+0.5 * d,
        sigma=sigma2,
    )

    return g1 + M * g2


# ============================================================
# 4. Analytic forward solution and synthetic observation
# ============================================================

@torch.no_grad()
def exact_scalar(x, y, t, M, R, d):
    """
    Analytic solution of:
        c_t + u(y)c_x = 0

    Since v = 0 and u depends only on y:
        c(x,y,t) = c0(x-u(y)t, y)
    """
    x_departure = x - shear_velocity(y) * t
    return initial_scalar(x_departure, y, M, R, d)


def make_synthetic_data(device, nx=NX, ny=NY):
    """
    Generate one synthetic X-ray-like image.

    The true parameters are used only here.
    They are NOT passed to the inverse model as initial guesses.
    """
    x_axis = torch.linspace(
        -LX / 2.0, LX / 2.0, nx + 1, device=device
    )[:-1]

    y_axis = torch.linspace(
        -LY / 2.0, LY / 2.0, ny + 1, device=device
    )[:-1]

    # Shape: (ny, nx)
    Y, X = torch.meshgrid(y_axis, x_axis, indexing="ij")

    M_true = torch.tensor(TRUE_PARAMS["M"], device=device)
    R_true = torch.tensor(TRUE_PARAMS["R"], device=device)
    d_true = torch.tensor(TRUE_PARAMS["d"], device=device)
    T_true = torch.tensor(TRUE_PARAMS["T"], device=device)

    c_true = exact_scalar(
        X.reshape(-1, 1),
        Y.reshape(-1, 1),
        T_true,
        M_true,
        R_true,
        d_true,
    ).reshape(ny, nx)

    # Simplified fixed-temperature X-ray emissivity proxy:
    # intensity proportional to density squared.
    intensity = c_true**2

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "X": X,
        "Y": Y,
        "c_true": c_true,
        "intensity": intensity,
    }


# ============================================================
# 5. Trainable bounded inverse parameters
# ============================================================

class InverseParameters(nn.Module):
    """
    Each physical parameter is represented as:
        p = p_min + (p_max-p_min)*sigmoid(raw_p)

    This keeps M, R, d, T inside their allowed intervals.
    """

    def __init__(self, initial_guess):
        super().__init__()

        self.names = ("M", "R", "d", "T")

        raw_values = []
        for name in self.names:
            lo, hi = PARAM_BOUNDS[name]
            z = (initial_guess[name] - lo) / (hi - lo)
            raw_values.append(inverse_sigmoid(z))

        self.raw = nn.Parameter(
            torch.tensor(raw_values, dtype=torch.float32)
        )

    def physical(self):
        values = {}

        for i, name in enumerate(self.names):
            lo, hi = PARAM_BOUNDS[name]
            values[name] = lo + (hi - lo) * torch.sigmoid(
                self.raw[i]
            )

        return values


def random_initial_guess():
    """Random starting point; never uses TRUE_PARAMS."""
    guess = {}

    for name, (lo, hi) in PARAM_BOUNDS.items():
        # Avoid starting too close to either bound.
        guess[name] = random.uniform(
            lo + 0.10 * (hi - lo),
            hi - 0.10 * (hi - lo),
        )

    return guess


# ============================================================
# 6. PINN architecture
# ============================================================

class ScalarPINN(nn.Module):
    """
    Inputs:
        x, y: physical coordinates
        tau: normalized time in [0,1]

    Output:
        c_theta(x,y,tau)

    Periodic Fourier embedding:
        sin/cos(2*pi*x/LX), sin/cos(2*pi*y/LY)

    Because only periodic features are used, the predicted scalar
    is periodic in x and y, including its spatial derivatives.

    Softplus makes c_theta nonnegative, appropriate for the
    density proxy used in this experiment.
    """

    def __init__(self, width=96, depth=4):
        super().__init__()

        layers = []
        input_dim = 5  # sin x, cos x, sin y, cos y, tau

        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        final_layer = nn.Linear(width, 1)

        # Start with a relatively small positive output after softplus.
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.constant_(final_layer.bias, -2.0)

        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        tau = coords[:, 2:3]

        features = torch.cat(
            [
                torch.sin(2.0 * math.pi * x / LX),
                torch.cos(2.0 * math.pi * x / LX),
                torch.sin(2.0 * math.pi * y / LY),
                torch.cos(2.0 * math.pi * y / LY),
                tau,
            ],
            dim=1,
        )

        raw_output = self.net(features)
        return F.softplus(raw_output, beta=1.0)


# ============================================================
# 7. PINN residuals and losses
# ============================================================

def predict_scalar(model, x, y, tau):
    coords = torch.cat([x, y, tau], dim=1)
    return model(coords)


def pde_residual(model, params, collocation):
    """
    In normalized time tau=t/T:
        c_tau + T*u(y)*c_x = 0
    """
    coords = collocation.detach().clone().requires_grad_(True)

    c = model(coords)

    grad_c = torch.autograd.grad(
        outputs=c,
        inputs=coords,
        grad_outputs=torch.ones_like(c),
        create_graph=True,
        retain_graph=True,
    )[0]

    c_x = grad_c[:, 0:1]
    c_tau = grad_c[:, 2:3]

    y = coords[:, 1:2]
    u = shear_velocity(y)

    T = params["T"]

    return c_tau + T * u * c_x


def compute_losses(
    model,
    inverse_params,
    collocation,
    ic_points,
    obs_coords,
    obs_intensity,
    c_scale,
    pde_scale,
    intensity_scale,
):
    """
    Returns total loss and individual components.

    Periodic boundary conditions are enforced by the network
    architecture, so there is no separate boundary penalty.
    """
    params = inverse_params.physical()

    # -------------------------
    # PDE loss
    # -------------------------
    residual = pde_residual(model, params, collocation)
    loss_pde = torch.mean((residual / pde_scale) ** 2)

    # -------------------------
    # Initial condition loss
    # -------------------------
    x_ic = ic_points[:, 0:1]
    y_ic = ic_points[:, 1:2]
    tau_ic = torch.zeros_like(x_ic)

    c_ic_pred = predict_scalar(model, x_ic, y_ic, tau_ic)

    c_ic_target = initial_scalar(
        x_ic,
        y_ic,
        params["M"],
        params["R"],
        params["d"],
    )

    loss_ic = torch.mean(
        ((c_ic_pred - c_ic_target) / c_scale) ** 2
    )

    # -------------------------
    # Observation loss at tau=1
    # -------------------------
    c_obs_pred = model(obs_coords)
    intensity_pred = c_obs_pred**2

    loss_obs = torch.mean(
        ((intensity_pred - obs_intensity) / intensity_scale) ** 2
    )

    # Equal weights are a starting point, not a universal optimum.
    loss_total = loss_pde + loss_ic + loss_obs

    components = {
        "total": loss_total,
        "pde": loss_pde,
        "ic": loss_ic,
        "obs": loss_obs,
    }

    return loss_total, components


def run_baseline_diagnostic(
    output_dir, model, params, device, pde_scale,
    nx=64, ny=64, nt=11, batch_size=4096,
):
    """Evaluate the trained PINN PDE residual on an independent dense grid.

    Reports raw residual statistics (not divided by pde_scale) globally and
    at each time slice, and saves spatial residual maps at selected times.
    This is a diagnostic only; it does not alter or retrain the model.
    """
    model.eval()
    x_axis = torch.linspace(-LX / 2.0, LX / 2.0, nx + 1, device=device)[:-1]
    y_axis = torch.linspace(-LY / 2.0, LY / 2.0, ny + 1, device=device)[:-1]
    tau_axis = torch.linspace(0.0, 1.0, nt, device=device)
    Y, X = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    residual_maps = []
    per_time = []
    all_abs_chunks = []
    all_sq_sum = 0.0
    all_count = 0
    global_max = 0.0

    # Evaluate one time slice at a time, in batches to limit memory use.
    for tau_value in tau_axis:
        tau_float = float(tau_value.detach().cpu())
        tau_flat = torch.full_like(x_flat, tau_float)
        coords_all = torch.cat([x_flat, y_flat, tau_flat], dim=1)
        residual_chunks = []

        for start in range(0, coords_all.shape[0], batch_size):
            coords = coords_all[start:start + batch_size].detach().clone()
            coords.requires_grad_(True)
            c = model(coords)
            grad_c = torch.autograd.grad(
                outputs=c,
                inputs=coords,
                grad_outputs=torch.ones_like(c),
                create_graph=False,
                retain_graph=False,
            )[0]
            c_x = grad_c[:, 0:1]
            c_tau = grad_c[:, 2:3]
            u = shear_velocity(coords[:, 1:2])
            residual = c_tau + params["T"] * u * c_x
            residual_chunks.append(residual.detach().reshape(-1))

        residual_slice = torch.cat(residual_chunks)
        abs_slice = residual_slice.abs()
        rms_slice = torch.sqrt(torch.mean(residual_slice ** 2)).item()
        mean_abs_slice = torch.mean(abs_slice).item()
        max_abs_slice = torch.max(abs_slice).item()
        pde_scale_value = float(pde_scale.detach().cpu()) if torch.is_tensor(pde_scale) else float(pde_scale)
        per_time.append({
            "tau": tau_float,
            "physical_time_using_estimated_T": tau_float * float(params["T"].detach().cpu()),
            "residual_rms": rms_slice,
            "residual_mean_abs": mean_abs_slice,
            "residual_max_abs": max_abs_slice,
            "normalized_residual_rms": rms_slice / max(pde_scale_value, 1e-30),
        })
        residual_maps.append(residual_slice.reshape(ny, nx).cpu().numpy())
        all_sq_sum += float(torch.sum(residual_slice ** 2).item())
        all_count += residual_slice.numel()
        global_max = max(global_max, max_abs_slice)
        all_abs_chunks.append(abs_slice.cpu())

    global_rms = math.sqrt(all_sq_sum / max(all_count, 1))
    all_abs = torch.cat(all_abs_chunks)
    global_mean_abs = float(all_abs.mean().item())
    global_median_abs = float(torch.quantile(all_abs, 0.50).item())
    global_p95_abs = float(torch.quantile(all_abs, 0.95).item())
    global_p99_abs = float(torch.quantile(all_abs, 0.99).item())
    pde_scale_value = float(pde_scale.detach().cpu()) if torch.is_tensor(pde_scale) else float(pde_scale)

    summary = {
        "diagnostic": "independent_dense_grid_PDE_residual",
        "grid": {"nx": nx, "ny": ny, "nt": nt, "points_total": nx * ny * nt},
        "estimated_parameters": {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in params.items()
        },
        "pde_scale": pde_scale_value,
        "raw_residual_global": {
            "rms": global_rms,
            "mean_absolute": global_mean_abs,
            "median_absolute": global_median_abs,
            "p95_absolute": global_p95_abs,
            "p99_absolute": global_p99_abs,
            "maximum_absolute": global_max,
        },
        "normalized_residual_global": {
            "rms": global_rms / max(pde_scale_value, 1e-30),
            "mean_absolute": global_mean_abs / max(pde_scale_value, 1e-30),
            "maximum_absolute": global_max / max(pde_scale_value, 1e-30),
        },
        "per_time_slice": per_time,
    }

    with (output_dir / "baseline_diagnostic.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (output_dir / "baseline_pde_residual_by_time.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(per_time[0].keys()))
        writer.writeheader()
        writer.writerows(per_time)

    # Plot raw residual statistics across normalized time.
    taus = np.asarray([row["tau"] for row in per_time])
    rms_values = np.asarray([row["residual_rms"] for row in per_time])
    max_values = np.asarray([row["residual_max_abs"] for row in per_time])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(taus, rms_values, marker="o")
    axes[0].set_title("PDE residual RMS by time")
    axes[0].set_xlabel(r"Normalized time $\tau$")
    axes[0].set_ylabel("Raw residual RMS")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(taus, max_values, marker="o")
    axes[1].set_title("Maximum absolute PDE residual by time")
    axes[1].set_xlabel(r"Normalized time $\tau$")
    axes[1].set_ylabel(r"$\max |r_\theta|$")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_pde_residual_statistics.png", dpi=160)
    plt.close(fig)

    # Spatial heatmaps at the first, middle, and final time slices.
    selected_indices = sorted(set([0, nt // 4, nt // 2, (3 * nt) // 4, nt - 1]))
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(3.4 * len(selected_indices), 3.5))
    if len(selected_indices) == 1:
        axes = [axes]
    extent = [-LX / 2.0, LX / 2.0, -LY / 2.0, LY / 2.0]
    vmax = max(float(np.max(np.abs(residual_maps[i]))) for i in selected_indices)
    vmax = max(vmax, 1e-12)
    image = None
    for ax, idx in zip(axes, selected_indices):
        image = ax.imshow(
            residual_maps[idx], origin="lower", extent=extent, aspect="auto",
            cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax.set_title(rf"$\tau={per_time[idx]['tau']:.2f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"Raw PDE residual $r_\theta$")
    fig.suptitle("Baseline diagnostic: PDE residual on independent dense grid")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_pde_residual_maps.png", dpi=160)
    plt.close(fig)

    print("\nBaseline PDE diagnostic (independent dense grid):")
    print(f"  Grid: {nx} x {ny} x {nt} = {nx * ny * nt:,} points")
    print(f"  Raw residual RMS:       {global_rms:.6e}")
    print(f"  Raw residual mean abs:  {global_mean_abs:.6e}")
    print(f"  Raw residual p95 abs:   {global_p95_abs:.6e}")
    print(f"  Raw residual p99 abs:   {global_p99_abs:.6e}")
    print(f"  Raw residual max abs:   {global_max:.6e}")
    print(f"  Normalized residual RMS: {global_rms / max(pde_scale_value, 1e-30):.6e}")
    print(f"  Saved diagnostics to: {output_dir}")
    return summary


# ============================================================
# 8. Plotting and saving
# ============================================================

def save_image_comparison(
    output_dir,
    data,
    model,
    device,
):
    model.eval()

    X = data["X"]
    Y = data["Y"]
    intensity = data["intensity"]

    ny, nx = intensity.shape

    tau = torch.ones(
        nx * ny, 1, device=device
    )

    obs_coords = torch.cat(
        [
            X.reshape(-1, 1),
            Y.reshape(-1, 1),
            tau,
        ],
        dim=1,
    )

    with torch.no_grad():
        c_pred = model(obs_coords).reshape(ny, nx)
        intensity_pred = c_pred**2

    error = torch.abs(intensity_pred - intensity)

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    obs_np = intensity.detach().cpu().numpy()
    pred_np = intensity_pred.detach().cpu().numpy()
    err_np = error.detach().cpu().numpy()

    # Save numerical arrays for later analysis.
    np.savez(
        output_dir / "image_comparison.npz",
        x=X_np,
        y=Y_np,
        intensity_observed=obs_np,
        intensity_predicted=pred_np,
        absolute_error=err_np,
    )

    extent = [
        -LX / 2.0,
        LX / 2.0,
        -LY / 2.0,
        LY / 2.0,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(
        obs_np,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    axes[0].set_title("Synthetic observation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        pred_np,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    axes[1].set_title("PINN reconstruction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_np,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(output_dir / "image_comparison.png", dpi=160)
    plt.close(fig)

    # Relative L2 error of intensity.
    rel_l2 = (
        torch.linalg.vector_norm(intensity_pred - intensity)
        / torch.linalg.vector_norm(intensity).clamp_min(1e-12)
    ).item()

    return rel_l2


def save_solution_animation(output_dir, data, model, device, estimated_params,
                            n_frames=61, fps=12):
    """Save a GIF comparing the PINN field with the analytic field.

    The analytic comparison uses the *estimated* M, R, d, and T. The PINN
    input uses normalized time tau in [0, 1], while physical time is t=tau*T.
    """
    model.eval()
    X, Y = data["X"], data["Y"]
    ny, nx = X.shape
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    M_est = torch.tensor(estimated_params["M"], device=device, dtype=X.dtype)
    R_est = torch.tensor(estimated_params["R"], device=device, dtype=X.dtype)
    d_est = torch.tensor(estimated_params["d"], device=device, dtype=X.dtype)
    T_est = torch.tensor(estimated_params["T"], device=device, dtype=X.dtype)
    tau_values = np.linspace(0.0, 1.0, n_frames)

    pinn_frames = []
    exact_frames = []
    with torch.no_grad():
        for tau_value in tau_values:
            tau_col = torch.full((nx * ny, 1), float(tau_value), device=device, dtype=X.dtype)
            coords = torch.cat([x_flat, y_flat, tau_col], dim=1)
            c_pinn = model(coords).reshape(ny, nx)
            c_exact = exact_scalar(
                x_flat, y_flat, T_est * float(tau_value), M_est, R_est, d_est
            ).reshape(ny, nx)
            pinn_frames.append(c_pinn.detach().cpu().numpy())
            exact_frames.append(c_exact.detach().cpu().numpy())

    max_value = max(float(np.max(a)) for a in pinn_frames + exact_frames)
    max_value = max(max_value, 1e-12)
    error_max = max(float(np.max(np.abs(a - b)))
                    for a, b in zip(pinn_frames, exact_frames))
    error_max = max(error_max, 1e-12)
    extent = [-LX / 2.0, LX / 2.0, -LY / 2.0, LY / 2.0]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    im0 = axes[0].imshow(pinn_frames[0], origin="lower", extent=extent,
                         aspect="auto", vmin=0.0, vmax=max_value, cmap="viridis")
    im1 = axes[1].imshow(exact_frames[0], origin="lower", extent=extent,
                         aspect="auto", vmin=0.0, vmax=max_value, cmap="viridis")
    im2 = axes[2].imshow(np.abs(pinn_frames[0] - exact_frames[0]), origin="lower",
                         extent=extent, aspect="auto", vmin=0.0, vmax=error_max,
                         cmap="magma")
    axes[0].set_title("PINN solution $c_\theta(x,y,\tau)$")
    axes[1].set_title("Analytic solution (estimated parameters)")
    axes[2].set_title("Absolute error")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    title = fig.suptitle(f"2D scalar solution, tau = {tau_values[0]:.3f}")

    def update(frame_idx):
        im0.set_data(pinn_frames[frame_idx])
        im1.set_data(exact_frames[frame_idx])
        im2.set_data(np.abs(pinn_frames[frame_idx] - exact_frames[frame_idx]))
        title.set_text(f"2D scalar solution, tau = {tau_values[frame_idx]:.3f} "
                       f"(physical time t = {tau_values[frame_idx] * estimated_params['T']:.3f})")
        return im0, im1, im2, title

    animation = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps,
                              blit=False)
    gif_path = output_dir / "solution_animation.gif"
    try:
        animation.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"Saved solution animation: {gif_path}")
    except (ImportError, RuntimeError) as exc:
        print(f"WARNING: Could not save GIF ({exc}). Install Pillow with: pip install pillow")
    finally:
        plt.close(fig)


def save_history_plots(output_dir, history, adam_epochs):
    """Plot losses and parameter estimates across both optimizers."""
    if not history:
        return

    steps = np.asarray([row["step"] for row in history], dtype=float)
    phases = np.asarray([row["phase"] for row in history])
    adam_mask = phases == "Adam"
    lbfgs_mask = phases == "L-BFGS"

    # Loss curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ("total", "pde", "ic", "obs"):
        values = np.asarray([max(float(row[name]), 1e-16) for row in history])
        if np.any(adam_mask):
            ax.plot(steps[adam_mask], values[adam_mask], label=f"{name} (Adam)")
        if np.any(lbfgs_mask):
            ax.plot(steps[lbfgs_mask], values[lbfgs_mask], label=f"{name} (L-BFGS)")

    ax.axvline(adam_epochs, color="gray", linestyle="--", linewidth=1.2,
               label="Adam / L-BFGS boundary")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative optimization step (Adam, then L-BFGS)")
    ax.set_ylabel("Normalized loss")
    ax.set_title("Training losses")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_history.png", dpi=160)
    plt.close(fig)

    # Parameter history
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.ravel()
    for ax, name in zip(axes, ("M", "R", "d", "T")):
        values = np.asarray([float(row[name]) for row in history])
        if np.any(adam_mask):
            ax.plot(steps[adam_mask], values[adam_mask], label="Adam estimate")
        if np.any(lbfgs_mask):
            ax.plot(steps[lbfgs_mask], values[lbfgs_mask], label="L-BFGS estimate")
        ax.axhline(TRUE_PARAMS[name], linestyle="--", label="synthetic truth")
        ax.axvline(adam_epochs, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Cumulative optimization step")
        ax.set_ylabel(name)
        ax.set_title(f"Parameter {name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "parameter_history.png", dpi=160)
    plt.close(fig)


def save_history_csv(output_dir, history):
    if not history:
        return

    path = output_dir / "training_history.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 9. One training run: Adam followed by L-BFGS
# ============================================================

def train_one_run(
    run_id,
    device,
    data,
    output_root,
    epochs=20000,
    n_f=2048,
    n_ic=2048,
    lbfgs_iter=2000,
    log_every=100,
    seed=1234,
    gif_frames=61,
    gif_fps=12,
    diagnostic_nx=64,
    diagnostic_ny=64,
    diagnostic_nt=11,
    diagnostic_batch_size=4096,
):
    set_seed(seed)

    output_dir = output_root / f"run_{run_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Independent random initial guess for this run.
    guess = random_initial_guess()

    inverse_params = InverseParameters(guess).to(device)
    model = ScalarPINN(width=96, depth=4).to(device)

    # Fixed collocation and IC samples make the L-BFGS phase stable.
    x_f, y_f = sample_uniform(n_f, device)
    tau_f = torch.rand(n_f, 1, device=device)
    collocation = torch.cat([x_f, y_f, tau_f], dim=1)

    x_ic, y_ic = sample_uniform(n_ic, device)
    ic_points = torch.cat([x_ic, y_ic], dim=1)

    # All observation pixels are used in the data loss.
    X = data["X"]
    Y = data["Y"]
    intensity = data["intensity"]

    ny, nx = intensity.shape
    tau_obs = torch.ones(nx * ny, 1, device=device)

    obs_coords = torch.cat(
        [
            X.reshape(-1, 1),
            Y.reshape(-1, 1),
            tau_obs,
        ],
        dim=1,
    )

    obs_intensity = intensity.reshape(-1, 1)

    # Scales are estimated from the synthetic observation,
    # not from the hidden true parameters.
    intensity_scale = obs_intensity.max().detach().clamp_min(1e-8)
    c_scale = torch.sqrt(intensity_scale).clamp_min(1e-8)

    # --------------------------------------------------------
    # Physically motivated nondimensionalization of the PDE loss
    #
    # PDE in normalized time tau = t/T:
    #     c_tau + T*u(y)*c_x = 0.
    #
    # Choose characteristic scales:
    #     c = C_ref * c_hat,   x = L_ref * x_hat,
    #     u = U0 * u_hat.
    #
    # Then:
    #     c_tau + T*u*c_x
    #       = C_ref * [c_hat_tau
    #                  + (T*U0/L_ref)*u_hat*c_hat_xhat].
    #
    # Both terms in the original residual have the same units
    # (the units of c), because tau is dimensionless. Therefore
    # the residual scale is C_ref only; do NOT multiply it by
    # (1 + T_ref*U0/L_ref), which would additionally weaken the
    # PDE loss without being required for dimensional consistency.
    #
    # C_ref is estimated from the observed intensity I ~ c^2:
    #     C_ref = sqrt(max(I_obs)).
    # L_ref = R1 is the fixed characteristic Gaussian width used
    # to interpret the dimensionless advection coefficient
    # beta = T*U0/L_ref. L_ref appears in the derivation, but
    # need not be multiplied into pde_scale in physical coords.
    # --------------------------------------------------------
    length_scale = R1
    velocity_scale = U0
    pde_scale = c_scale

    print(f"Characteristic concentration scale C_ref = {float(c_scale.item()):.6e}")
    print(f"Characteristic length scale L_ref = {length_scale:.6e}")
    print(f"Characteristic velocity scale U0 = {velocity_scale:.6e}")
    print(
        "Dimensionless advection coefficient beta=T*U0/L_ref "
        f"at T-bound midpoint = "
        f"{0.5 * (PARAM_BOUNDS['T'][0] + PARAM_BOUNDS['T'][1]) * velocity_scale / length_scale:.6e}"
    )
    print(f"New PDE residual scale pde_scale = {float(pde_scale.item()):.6e}")

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": 1.0e-3},
            {"params": inverse_params.parameters(), "lr": 3.0e-3},
        ]
    )

    history = []
    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"Run {run_id}: initial guess = {guess}")
    print(f"Device = {device}")
    print("=" * 70)

    # -------------------------
    # Adam
    # -------------------------
    for step in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total, losses = compute_losses(
            model=model,
            inverse_params=inverse_params,
            collocation=collocation,
            ic_points=ic_points,
            obs_coords=obs_coords,
            obs_intensity=obs_intensity,
            c_scale=c_scale,
            pde_scale=pde_scale,
            intensity_scale=intensity_scale,
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(inverse_params.parameters()),
            max_norm=100.0,
        )
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == epochs:
            physical = inverse_params.physical()

            row = {
                "phase": "Adam",
                "phase_step": step,
                "step": step,
                "total": losses["total"].detach().item(),
                "pde": losses["pde"].detach().item(),
                "ic": losses["ic"].detach().item(),
                "obs": losses["obs"].detach().item(),
                "M": physical["M"].detach().item(),
                "R": physical["R"].detach().item(),
                "d": physical["d"].detach().item(),
                "T": physical["T"].detach().item(),
            }
            history.append(row)

            print(
                f"Adam {step:5d}/{epochs} | "
                f"L={row['total']:.3e} "
                f"PDE={row['pde']:.3e} "
                f"IC={row['ic']:.3e} "
                f"Obs={row['obs']:.3e} | "
                f"M={row['M']:.4f} "
                f"R={row['R']:.4f} "
                f"d={row['d']:.4f} "
                f"T={row['T']:.4f}"
            )

    # -------------------------
    # L-BFGS refinement
    # -------------------------
    print("\nStarting L-BFGS refinement...")

    # Calling LBFGS.step(closure) once per outer iteration lets us record
    # post-step losses/parameters. The optimizer state (including curvature
    # history) is retained across calls, so this remains one continuous run.
    lbfgs = torch.optim.LBFGS(
        list(model.parameters()) + list(inverse_params.parameters()),
        lr=0.8,
        max_iter=1,
        max_eval=20,
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        total, _ = compute_losses(
            model=model,
            inverse_params=inverse_params,
            collocation=collocation,
            ic_points=ic_points,
            obs_coords=obs_coords,
            obs_intensity=obs_intensity,
            c_scale=c_scale,
            pde_scale=pde_scale,
            intensity_scale=intensity_scale,
        )
        total.backward()
        return total

    for lbfgs_step in range(1, lbfgs_iter + 1):
        model.train()
        lbfgs.step(closure)

        # Evaluate at the updated parameters (not at a line-search trial point).
        model.eval()
        with torch.enable_grad():
            _, losses = compute_losses(
                model=model,
                inverse_params=inverse_params,
                collocation=collocation,
                ic_points=ic_points,
                obs_coords=obs_coords,
                obs_intensity=obs_intensity,
                c_scale=c_scale,
                pde_scale=pde_scale,
                intensity_scale=intensity_scale,
            )

        physical = inverse_params.physical()
        row = {
            "phase": "L-BFGS",
            "phase_step": lbfgs_step,
            "step": epochs + lbfgs_step,
            "total": losses["total"].detach().item(),
            "pde": losses["pde"].detach().item(),
            "ic": losses["ic"].detach().item(),
            "obs": losses["obs"].detach().item(),
            "M": physical["M"].detach().item(),
            "R": physical["R"].detach().item(),
            "d": physical["d"].detach().item(),
            "T": physical["T"].detach().item(),
        }
        history.append(row)

        if (lbfgs_step == 1 or lbfgs_step % log_every == 0
                or lbfgs_step == lbfgs_iter):
            print(
                f"L-BFGS {lbfgs_step:5d}/{lbfgs_iter} | "
                f"L={row['total']:.3e} "
                f"PDE={row['pde']:.3e} "
                f"IC={row['ic']:.3e} "
                f"Obs={row['obs']:.3e} | "
                f"M={row['M']:.4f} "
                f"R={row['R']:.4f} "
                f"d={row['d']:.4f} "
                f"T={row['T']:.4f}"
            )

    # Final loss evaluation
    model.eval()
    with torch.enable_grad():
        final_loss, final_components = compute_losses(
            model=model,
            inverse_params=inverse_params,
            collocation=collocation,
            ic_points=ic_points,
            obs_coords=obs_coords,
            obs_intensity=obs_intensity,
            c_scale=c_scale,
            pde_scale=pde_scale,
            intensity_scale=intensity_scale,
        )

    final_params = {
        name: value.detach().item()
        for name, value in inverse_params.physical().items()
    }

    # Baseline diagnostic: evaluate PDE residual away from training collocation points.
    diagnostic_summary = run_baseline_diagnostic(
        output_dir=output_dir,
        model=model,
        params=inverse_params.physical(),
        device=device,
        pde_scale=pde_scale,
        nx=diagnostic_nx,
        ny=diagnostic_ny,
        nt=diagnostic_nt,
        batch_size=diagnostic_batch_size,
    )

    elapsed = time.time() - start_time

    print("\nFinal estimated parameters:")
    for name in ("M", "R", "d", "T"):
        print(
            f"  {name}: estimate={final_params[name]:.6f}, "
            f"synthetic truth={TRUE_PARAMS[name]:.6f}"
        )

    print(f"Final normalized loss: {final_loss.detach().item():.6e}")
    print(f"Wall-clock time: {elapsed:.1f} seconds")

    # Save model and parameter state.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "inverse_params_state_dict": inverse_params.state_dict(),
            "estimated_params": final_params,
            "initial_guess": guess,
            "seed": seed,
        },
        output_dir / "checkpoint.pt",
    )

    # Save final loss components.
    final_record = {
        "run_id": run_id,
        "seed": seed,
        **{f"estimate_{k}": v for k, v in final_params.items()},
        **{f"truth_{k}": v for k, v in TRUE_PARAMS.items()},
        **{
            f"loss_{k}": v.detach().item()
            for k, v in final_components.items()
        },
        "relative_intensity_L2": None,
        "wall_clock_seconds": elapsed,
        "initial_guess": json.dumps(guess),
        "baseline_diagnostic_global_raw_residual_rms": diagnostic_summary["raw_residual_global"]["rms"],
        "baseline_diagnostic_global_raw_residual_max_abs": diagnostic_summary["raw_residual_global"]["maximum_absolute"],
    }

    rel_l2 = save_image_comparison(
        output_dir=output_dir,
        data=data,
        model=model,
        device=device,
    )

    final_record["relative_intensity_L2"] = rel_l2

    with (output_dir / "final_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(final_record, f, indent=2, ensure_ascii=False)

    save_history_plots(output_dir, history, adam_epochs=epochs)
    save_history_csv(output_dir, history)
    save_solution_animation(
        output_dir=output_dir,
        data=data,
        model=model,
        device=device,
        estimated_params=final_params,
        n_frames=gif_frames,
        fps=gif_fps,
    )

    return final_record


# ============================================================
# 10. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--n_f", type=int, default=2048)
    parser.add_argument("--n_ic", type=int, default=2048)
    parser.add_argument("--lbfgs_iter", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--gif_frames", type=int, default=61, help="Number of frames in each solution GIF")
    parser.add_argument("--gif_fps", type=int, default=12, help="Frame rate of each solution GIF")
    parser.add_argument("--diagnostic_nx", type=int, default=64, help="Dense-grid x resolution for PDE baseline diagnostic")
    parser.add_argument("--diagnostic_ny", type=int, default=64, help="Dense-grid y resolution for PDE baseline diagnostic")
    parser.add_argument("--diagnostic_nt", type=int, default=11, help="Number of normalized-time slices for PDE baseline diagnostic")
    parser.add_argument("--diagnostic_batch_size", type=int, default=4096, help="Batch size for PDE residual diagnostic")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str, default="results_pinn")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # Generate one synthetic observation, shared by all restarts.
    data = make_synthetic_data(
        device=device,
        nx=args.resolution,
        ny=args.resolution,
    )

    # Save the synthetic observation separately.
    np.savez(
        output_root / "synthetic_observation.npz",
        X=data["X"].detach().cpu().numpy(),
        Y=data["Y"].detach().cpu().numpy(),
        intensity=data["intensity"].detach().cpu().numpy(),
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        data["intensity"].detach().cpu().numpy(),
        origin="lower",
        extent=[
            -LX / 2.0, LX / 2.0,
            -LY / 2.0, LY / 2.0,
        ],
        aspect="auto",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic X-ray-like observation")
    fig.colorbar(im, ax=ax, label="Intensity")
    fig.tight_layout()
    fig.savefig(output_root / "synthetic_observation.png", dpi=160)
    plt.close(fig)

    print("\nSynthetic data generated.")
    print("True parameters are used only for data generation:")
    print(json.dumps(TRUE_PARAMS, indent=2))

    results = []

    for run_id in range(1, args.restarts + 1):
        result = train_one_run(
            run_id=run_id,
            device=device,
            data=data,
            output_root=output_root,
            epochs=args.epochs,
            n_f=args.n_f,
            n_ic=args.n_ic,
            lbfgs_iter=args.lbfgs_iter,
            seed=args.seed + run_id,
            gif_frames=args.gif_frames,
            gif_fps=args.gif_fps,
            diagnostic_nx=args.diagnostic_nx,
            diagnostic_ny=args.diagnostic_ny,
            diagnostic_nt=args.diagnostic_nt,
            diagnostic_batch_size=args.diagnostic_batch_size,
        )
        results.append(result)

    # Save a summary table across random restarts.
    summary_path = output_root / "summary.csv"

    if results:
        with summary_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 70)
    print("All runs completed.")
    print(f"Results saved to: {output_root.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
