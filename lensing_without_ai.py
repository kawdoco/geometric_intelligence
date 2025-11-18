
from __future__ import annotations
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
N = 256
L = 1.0
NOISE_STD = 0.02
SEED = 42
#SAVE_DIR = "/data"  # save into notebook sandbox by default

import os
SAVE_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

def make_grid(N=256, L=1.0):
    d = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')
    kx = 2*np.pi*fftfreq(N, d=d)
    ky = 2*np.pi*fftfreq(N, d=d)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    return X, Y, KX, KY, K2

def gaussian_2d(X, Y, xc, yc, amp=1.0, sx=0.06, sy=None):
    if sy is None: sy = sx
    return amp * np.exp(-0.5 * ((X-xc)**2/sx**2 + (Y-yc)**2/sy**2))

def poisson_solve_from_kappa(kappa, K2):
    kappa_hat = fft2(kappa)
    psi_hat = np.zeros_like(kappa_hat, dtype=complex)
    mask = K2 != 0
    psi_hat[mask] = -2.0 * kappa_hat[mask] / K2[mask]
    psi = np.real(ifft2(psi_hat))
    return psi, psi_hat

def shear_from_psi_hat(psi_hat, KX, KY):
    psi_xx_hat = -(KX**2) * psi_hat
    psi_yy_hat = -(KY**2) * psi_hat
    psi_xy_hat = -(KX*KY) * psi_hat
    gamma1_hat = 0.5 * (psi_xx_hat - psi_yy_hat)
    gamma2_hat = psi_xy_hat
    gamma1 = np.real(ifft2(gamma1_hat))
    gamma2 = np.real(ifft2(gamma2_hat))
    return gamma1, gamma2

def kaiser_squires(gamma1, gamma2, KX, KY):
    g = gamma1 + 1j*gamma2
    g_hat = fft2(g)
    K2 = KX**2 + KY**2
    D = np.zeros_like(g_hat, dtype=complex)
    mask = K2 != 0
    D[mask] = ((KX[mask]**2 - KY[mask]**2) + 2j*KX[mask]*KY[mask]) / K2[mask]
    kappa_hat = np.conj(D) * g_hat
    kappa = np.real(ifft2(kappa_hat))
    return kappa

def basic_metrics(truth, pred):
    t = truth.flatten()
    p = pred.flatten()
    mse = float(np.mean((t - p)**2))
    mae = float(np.mean(np.abs(t - p)))
    t0 = t - t.mean()
    p0 = p - p.mean()
    denom = (np.sqrt((t0**2).sum()) * np.sqrt((p0**2).sum()) + 1e-12)
    corr = float((t0*p0).sum() / denom)
    return mse, mae, corr

def save_field_image(field, title, subtitle, filename):
    plt.figure(figsize=(6,6))
    plt.imshow(field, origin='lower', interpolation='nearest')
    plt.title(title, pad=12)
    plt.text(4, field.shape[0]-10, subtitle, fontsize=10, bbox=dict(boxstyle="round", alpha=0.3))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

def save_shear_quiver(g1, g2, step, title, subtitle, filename):
    H, W = g1.shape
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    Xs, Ys = np.meshgrid(xs, ys, indexing='xy')
    U = g1[Ys, Xs]
    V = g2[Ys, Xs]
    plt.figure(figsize=(6,6))
    plt.quiver(Xs, Ys, U, V, angles='xy', scale_units='xy')
    plt.gca().invert_yaxis()
    plt.title(title, pad=12)
    plt.text(4, 10, subtitle, fontsize=10, bbox=dict(boxstyle="round", alpha=0.3))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

def save_kappa_with_streamlines(kappa, psi, filename, title="OUTPUT: κ̂ with Streamlines", subtitle="Streamlines from ∇ψ̂"):
    gy, gx = np.gradient(psi)
    plt.figure(figsize=(6,6))
    plt.imshow(kappa, origin='lower', interpolation='nearest', alpha=0.9)
    Y, X = np.mgrid[0:kappa.shape[0], 0:kappa.shape[1]]
    step = 4
    U = gx[::step, ::step]
    V = gy[::step, ::step]
    XX = X[::step, ::step]
    YY = Y[::step, ::step]
    try:
        plt.streamplot(XX, YY, U, V, density=1.3, linewidth=1)
    except Exception:
        pass
    plt.title(title, pad=12)
    plt.text(4, kappa.shape[0]-10, subtitle, fontsize=10, bbox=dict(boxstyle="round", alpha=0.3))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    X, Y, KX, KY, K2 = make_grid(N=N, L=L)

    # Synthetic κ (sum of a few Gaussians)
    kappa_true = (
        gaussian_2d(X, Y, 0.35, 0.55, amp=1.2, sx=0.06) +
        gaussian_2d(X, Y, 0.70, 0.40, amp=0.9,  sx=0.08) +
        0.15 * gaussian_2d(X, Y, 0.55, 0.75, amp=1.0, sx=0.05)
    )
    kappa_true -= 0.2 * kappa_true.mean()

    # ψ and shear
    psi_true, psi_true_hat = poisson_solve_from_kappa(kappa_true, K2)
    gamma1_true, gamma2_true = shear_from_psi_hat(psi_true_hat, KX, KY)

    # Add noise
    rng = np.random.default_rng(42)
    gamma1_obs = gamma1_true + NOISE_STD * rng.standard_normal((N, N))
    gamma2_obs = gamma2_true + NOISE_STD * rng.standard_normal((N, N))

    # Reconstruct κ via Kaiser–Squires
    kappa_rec = kaiser_squires(gamma1_obs, gamma2_obs, KX, KY)
    psi_rec, _ = poisson_solve_from_kappa(kappa_rec, K2)

    # Metrics
    mse, mae, corr = basic_metrics(kappa_true, kappa_rec)
    print("==== Reconstruction metrics (κ̂ vs κ) ====")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Corr: {corr:.4f}")
    print("=========================================")

    # Save images
    save_field_image(gamma1_obs, "INPUT: Shear γ1", "Averaged galaxy shapes (with noise)", f"{SAVE_DIR}/input_gamma1.png")
    save_field_image(gamma2_obs, "INPUT: Shear γ2", "Averaged galaxy shapes (with noise)", f"{SAVE_DIR}/input_gamma2.png")
    save_shear_quiver(gamma1_obs, gamma2_obs, step=8, title="INPUT: Shear Field (sticks)",
                      subtitle="Orientation & length indicate local distortion", filename=f"{SAVE_DIR}/input_shear_sticks.png")
    save_field_image(kappa_true, "GROUND TRUTH: κ (curvature)", "Synthetic mass-induced curvature", f"{SAVE_DIR}/kappa_truth.png")
    save_field_image(kappa_rec, "OUTPUT: κ̂ (reconstructed)", f"MSE={mse:.4f}, MAE={mae:.4f}, Corr={corr:.3f}", f"{SAVE_DIR}/kappa_reconstructed.png")
    save_kappa_with_streamlines(kappa_rec, psi_rec, filename=f"{SAVE_DIR}/kappa_streamlines.png")

if __name__ == "__main__":
    main()
