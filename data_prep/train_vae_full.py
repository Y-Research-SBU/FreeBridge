"""Train the convolutional VAE used by the BBBC021 latent pipeline.

Checkpoint format:
    {"model": state_dict, "zdim": int, "img_size": int}
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# --------------------------- model (matches export_latents_mu.py) ---------------------------
class Enc(nn.Module):
    """Conv encoder: (B,3,96,96) -> (mu, logvar), each (B, zdim)."""
    def __init__(self, zdim: int = 64, img_size: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  nn.ReLU(inplace=True),   # 96 -> 48
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 48 -> 24
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True), # 24 -> 12
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(inplace=True), # 12 -> 6
        )
        feat_hw = img_size // 16            # 96 -> 6
        self.flat = 512 * feat_hw * feat_hw
        self.fc_mu = nn.Linear(self.flat, zdim)
        self.fc_logvar = nn.Linear(self.flat, zdim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Dec(nn.Module):
    """Decoder: (B, zdim) -> (B,3,96,96) in [0,1]."""
    def __init__(self, zdim: int = 64, img_size: int = 96):
        super().__init__()
        self.in_hw = img_size // 16         # 6
        self.in_ch = 512
        self.fc = nn.Linear(zdim, self.in_ch * self.in_hw * self.in_hw)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True),  # 6 -> 12
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 12 -> 24
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(inplace=True),  # 24 -> 48
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    nn.Sigmoid(),           # 48 -> 96
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.in_ch, self.in_hw, self.in_hw)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, zdim: int = 64, img_size: int = 96):
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError(f"img_size must be divisible by 16 (4x stride-2 down/up-sampling), got {img_size}")
        self.enc = Enc(zdim, img_size)
        self.dec = Dec(zdim, img_size)

    def forward(self, x):
        mu, logvar = self.enc(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar


# --------------------------- data ---------------------------
class CropDataset(Dataset):
    def __init__(self, root: str, img_size: int = 96):
        exts = ["png", "jpg", "jpeg", "tif", "tiff"]
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
            paths += glob.glob(os.path.join(root, f"**/*.{e.upper()}"), recursive=True)
        self.paths = sorted(set(paths))
        if not self.paths:
            raise RuntimeError(f"No images under {root}")
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()


# --------------------------- loss + train ---------------------------
def vae_loss(xhat, x, mu, logvar, beta: float):
    recon = F.mse_loss(xhat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon.item(), kld.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(args.zdim, args.img_size).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    ds = CropDataset(args.crops, args.img_size)
    if len(ds) < args.batch:
        raise ValueError(f"Dataset has {len(ds)} images but batch={args.batch} with drop_last=True; reduce --batch.")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        vae.train()
        for x in dl:
            x = x.to(device, non_blocking=True)
            xhat, mu, logvar = vae(x)
            loss, r, k = vae_loss(xhat, x, mu, logvar, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f} | recon {r:.4f} | kl {k:.4f}")
            step += 1
        torch.save({"model": vae.state_dict(), "zdim": args.zdim, "img_size": args.img_size}, args.out)
    print(f"saved VAE checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
