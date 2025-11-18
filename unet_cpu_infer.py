
# unet_cpu_infer.py
import os, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(max(1, os.cpu_count()//2))

def make_grid(N=128, L=1.0):
    d = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')
    kx = 2*np.pi*np.fft.fftfreq(N, d=d)
    ky = 2*np.pi*np.fft.fftfreq(N, d=d)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    return X, Y, KX, KY, K2

def gaussian_2d(X, Y, xc, yc, amp=1.0, sx=0.06, sy=None):
    if sy is None: sy = sx
    return amp * np.exp(-0.5 * ((X-xc)**2/sx**2 + (Y-yc)**2/sy**2))

def poisson_solve_from_kappa(kappa, K2):
    K = np.fft.fft2(kappa)
    psi_hat = np.zeros_like(K, dtype=complex)
    m = (K2 != 0)
    psi_hat[m] = -2.0 * K[m] / K2[m]
    psi = np.real(np.fft.ifft2(psi_hat))
    return psi, psi_hat

def shear_from_psi_hat(psi_hat, KX, KY):
    psi_xx_hat = -(KX**2) * psi_hat
    psi_yy_hat = -(KY**2) * psi_hat
    psi_xy_hat = -(KX*KY) * psi_hat
    g1_hat = 0.5 * (psi_xx_hat - psi_yy_hat)
    g2_hat = psi_xy_hat
    g1 = np.real(np.fft.ifft2(g1_hat))
    g2 = np.real(np.fft.ifft2(g2_hat))
    return g1, g2

def make_sample(N=128, L=1.0, noise_std=0.02, seed=999):
    rng = np.random.default_rng(seed)
    X, Y, KX, KY, K2 = make_grid(N, L)
    n_halo = rng.integers(2, 5)
    kappa = np.zeros((N, N), dtype=np.float32)
    for _ in range(n_halo):
        xc = rng.uniform(0.2, 0.8)
        yc = rng.uniform(0.2, 0.8)
        amp = rng.uniform(0.6, 1.4)
        sx  = rng.uniform(0.04, 0.10)
        kappa += gaussian_2d(X, Y, xc, yc, amp=amp, sx=sx).astype(np.float32)
    kappa -= 0.2 * float(kappa.mean())
    psi, psi_hat = poisson_solve_from_kappa(kappa, K2)
    g1, g2 = shear_from_psi_hat(psi_hat, KX, KY)
    g1n = g1 + noise_std * rng.standard_normal((N, N))
    g2n = g2 + noise_std * rng.standard_normal((N, N))
    x = np.stack([g1n, g2n], axis=0).astype(np.float32)
    y = kappa[None, ...].astype(np.float32)
    return x, y

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=8, depth=3):
        super().__init__()
        chs = [base*(2**i) for i in range(depth)]
        self.enc1 = DoubleConv(in_ch, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.pool = nn.MaxPool2d(2)
        self.bot  = DoubleConv(chs[2], chs[2]*2)
        self.up3  = nn.ConvTranspose2d(chs[2]*2, chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[2]*2, chs[2])
        self.up2  = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[1]*2, chs[1])
        self.up1  = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[0]*2, chs[0])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.up3(b); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1)

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    model = TinyUNet(in_ch=2, out_ch=1, base=args.base, depth=args.depth)
    wpath = os.path.join(args.outdir, "tiny_unet_cpu.pth")
    if not os.path.exists(wpath):
        print(f"ERROR: weights not found at {wpath}. Train first with unet_cpu_train.py")
        return
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    x, y = make_sample(N=args.size, noise_std=args.noise, seed=999)
    xb = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
    plt.figure(figsize=(9,3.5))
    plt.subplot(1,3,1); plt.imshow(x[0], origin='lower'); plt.title("γ1"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(x[1], origin='lower'); plt.title("γ2"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(pred, origin='lower'); plt.title("κ̂ (pred)"); plt.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "infer_pred.png"), dpi=170); plt.close()
    print(f"Saved prediction to {args.outdir}/infer_pred.png")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="outputs_unet")
    return ap.parse_args()

if __name__ == "__main__":
    main(parse_args())
