import argparse
import os
from pathlib import Path
import json

import numpy as np
import torch

try:
    from .cellvae_adapter import CellVAEAdapter, CellVAEAdapterConfig
except ImportError:
    from cellvae_adapter import CellVAEAdapter, CellVAEAdapterConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="Path to latents .npy with shape (N,D)")
    ap.add_argument("--vae-ckpt", required=True, help="Path to VAE checkpoint")
    ap.add_argument("--model-import", required=True, help="e.g. train_vae_full:VAE (run from repo root: python data_prep/gen_cells.py ...)")
    ap.add_argument("--model-kwargs", default="{}", help="JSON kwargs for model constructor")
    ap.add_argument("--image-size", nargs=3, type=int, required=True, metavar=("C", "H", "W"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=64, help="How many images to generate (<=N)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--save-individual", action="store_true", help="Also save each image as png")
    ap.add_argument("--allow-partial-load", action="store_true", help="allow partial VAE load (default: strict)")
    ap.add_argument("--out-range", default="auto", choices=["auto", "0_1", "neg1_1"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat = np.load(args.latents)
    if lat.ndim != 2:
        raise RuntimeError(f"latents must be (N,D), got shape={lat.shape}")
    N, D = lat.shape
    if N == 0:
        raise RuntimeError(f"No latents in {args.latents}")
    n = min(args.n, N)
    print(f"[Latents] {args.latents} shape=(N,D)=({N},{D}) using n={n}")

    torch.manual_seed(args.seed)
    idx = np.random.RandomState(args.seed).choice(N, size=n, replace=False)
    z_np = lat[idx]
    z = torch.from_numpy(z_np).float()

    cfg = CellVAEAdapterConfig(
        ckpt_path=args.vae_ckpt,
        model_import=args.model_import,
        model_kwargs_json=args.model_kwargs,
        image_size=tuple(args.image_size),
        device=args.device,
        strict=not args.allow_partial_load,
        out_range=args.out_range,
    )
    vae = CellVAEAdapter(cfg)

    xs = []
    with torch.no_grad():
        for i in range(0, n, args.batch):
            zb = z[i : i + args.batch]
            xb = vae.decode(zb)
            xb = vae.to_01(xb).cpu()
            xs.append(xb)

    x = torch.cat(xs, dim=0)  # (n,C,H,W)
    print(f"[Decode] decoded images: {tuple(x.shape)} range=({x.min().item():.4f},{x.max().item():.4f})")

    import torchvision.utils as vutils  # lazy: keep --help working without torchvision
    grid = vutils.make_grid(x, nrow=args.nrow, pad_value=1.0)
    grid_path = out_dir / "grid.png"
    vutils.save_image(grid, str(grid_path))
    print(f"[Save] grid: {grid_path}")

    if args.save_individual:
        indiv_dir = out_dir / "indiv"
        indiv_dir.mkdir(exist_ok=True)
        for k in range(n):
            vutils.save_image(x[k], str(indiv_dir / f"{k:05d}.png"))
        print(f"[Save] individual: {indiv_dir}")


if __name__ == "__main__":
    main()