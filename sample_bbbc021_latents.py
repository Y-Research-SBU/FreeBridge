#!/usr/bin/env python3
"""Sample generated target latents from a trained FreeBridge checkpoint.

Closes the loop: trained bridge checkpoint -> forward-sampled target latents.npy,
which can then be decoded to cell images with data_prep/gen_cells.py.

The checkpoint must keep its sibling `.hydra/config.yaml` (Hydra run directory),
since restore_model reads the config from there.

Example:
    python sample_bbbc021_latents.py \
        --ckpt outputs/runs/bbbc021/<date>/<time>/checkpoints/last.ckpt \
        --out_npy outputs/generated_latents.npy --n 1024
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained .ckpt (with sibling .hydra/config.yaml)")
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--n", type=int, default=1024, help="number of source samples to transport")
    ap.add_argument("--nfe", type=int, default=None, help="integration steps (defaults to cfg.nfe)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=["train", "val"], default="val",
                    help="which source distribution to start from")
    args = ap.parse_args()
    if args.n <= 0:
        raise ValueError(f"--n must be positive, got {args.n}")

    from pathlib import Path
    from freebridge.utils import restore_model

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model, cfg = restore_model(args.ckpt, device=device)
    model.eval()

    # restore_model already built the samplers; reuse them via get_dist_boundary
    from freebridge.dataset import get_dist_boundary
    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    source = p0_val if args.split == "val" else p0

    xinit = source(args.n).to(device)
    nfe = int(args.nfe or cfg.nfe)
    log_steps = int(cfg.csoc.T_mean)
    if nfe + 1 < log_steps:
        raise ValueError(f"nfe+1 must be >= csoc.T_mean; got nfe={nfe}, T_mean={log_steps}")
    with torch.no_grad():
        out = model.sample(xinit, log_steps=log_steps, direction="fwd", nfe=nfe)

    z_gen = out["xs"][:, -1].detach().cpu().numpy()
    out_path = Path(args.out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, z_gen)
    print(f"[OK] generated latents {z_gen.shape} -> {args.out_npy}")


if __name__ == "__main__":
    main()
