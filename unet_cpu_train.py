
# unet_cpu_train.py
import os, math, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

def make_sample(N=128, L=1.0, noise_std=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng()
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

class LensingSyntheticDS(Dataset):
    def __init__(self, n_samples=512, N=128, noise_std=0.02, seed=13):
        self.n_samples = n_samples; self.N = N; self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((n_samples, 2, N, N), dtype=np.float32)
        self.Y = np.zeros((n_samples, 1, N, N), dtype=np.float32)
        for i in range(n_samples):
            x, y = make_sample(N=N, noise_std=noise_std, rng=self.rng)
            self.X[i] = x; self.Y[i] = y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # <-- add these three lines
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        elif isinstance(idx, np.integer):
            idx = int(idx)
        # --
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

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

def train(args):
    device = torch.device("cpu")
    os.makedirs(args.outdir, exist_ok=True)
    ds = LensingSyntheticDS(n_samples=args.samples, N=args.size, noise_std=args.noise, seed=13)
    n_train = int(0.9 * len(ds)); n_val = len(ds) - n_train
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = TinyUNet(in_ch=2, out_ch=1, base=args.base, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    best_val = float('inf'); tr_hist=[]; va_hist=[]
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train(); tr_loss=0.0
        for xb, yb in train_dl:
            pred = model(xb.to(device))
            loss = loss_fn(pred, yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_dl.dataset)

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb.to(device))
                va_loss += loss_fn(pred, yb.to(device)).item() * xb.size(0)
        va_loss /= len(val_dl.dataset)
        tr_hist.append(tr_loss); va_hist.append(va_loss)
        print(f"Epoch {ep:02d}/{args.epochs}  train L1={tr_loss:.4f}  val L1={va_loss:.4f}")
        if va_loss < best_val:
            best_val=va_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "tiny_unet_cpu.pth"))
    print(f"Done in {time.time()-t0:.1f}s. Best val L1={best_val:.4f}. Weights saved.")

    # Plot curves
    plt.figure(figsize=(6,4))
    plt.plot(tr_hist, label="train L1"); plt.plot(va_hist, label="val L1")
    plt.xlabel("epoch"); plt.ylabel("L1"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "train_curve.png"), dpi=160); plt.close()

    # Sample prediction
    model.load_state_dict(torch.load(os.path.join(args.outdir, "tiny_unet_cpu.pth"), map_location="cpu"))
    model.eval()
    xb, yb = ds[0]
    with torch.no_grad():
        pred = model(xb.unsqueeze(0)).squeeze(0).squeeze(0).cpu().numpy()
    gt = yb.squeeze(0).numpy()
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(gt, origin="lower"); plt.title("Target κ"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(pred, origin="lower"); plt.title("Pred κ̂"); plt.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "sample_train_pred.png"), dpi=160); plt.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="outputs_unet")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
