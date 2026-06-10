#!/usr/bin/env python3
"""Build control/perturbed latent endpoint arrays for the FreeBridge config.

Takes the per-crop latents exported by `export_latents_from_vae.py` (which saves
{"Z": (N, D) tensor, "paths": [crop_path, ...]}) and splits them into the
`src_*` (control) and `tgt_*` (perturbed) `.npy` arrays referenced in
`configs/experiment/bbbc021.yaml`.

The control vs. perturbed assignment is decided by matching each crop's source
path against user-provided regexes. IMPORTANT: BBBC021 crop filenames follow the
plate/well/site convention (e.g. "Week1_150607_B02_s1_c124_cell0001.png") and do
NOT contain compound / MoA names. The regexes therefore must match plate/well/site
identifiers, OR you must first organize crops into condition-named folders, OR map
plate/well/site -> compound via the BBBC021 metadata before running this script.
There is no universal default; the grouping is dataset specific.

Example:
    python data_prep/build_bbbc021_endpoints.py \
        --train_pt exports/train_latents.pt \
        --val_pt   exports/val_latents.pt \
        --control_regex "DMSO|control" \
        --target_regex  "taxol|Taxol" \
        --out_dir data/bbbc021
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch


def _split(pt_path: str, control_re, target_re):
    blob = torch.load(pt_path, map_location="cpu")
    Z = blob["Z"]
    Z = Z.numpy() if hasattr(Z, "numpy") else np.asarray(Z)
    paths = blob["paths"]
    if Z.ndim != 2 or len(paths) != Z.shape[0]:
        raise ValueError(f"Bad latents file {pt_path}: Z {Z.shape}, paths {len(paths)}")

    src_idx, tgt_idx = [], []
    for i, p in enumerate(paths):
        name = str(p)
        is_ctrl = control_re.search(name) is not None
        is_tgt = target_re.search(name) is not None
        if is_ctrl and is_tgt:
            raise ValueError(f"Path matches BOTH control and target regex: {name}")
        elif is_ctrl:
            src_idx.append(i)
        elif is_tgt:
            tgt_idx.append(i)
    if not src_idx or not tgt_idx:
        raise ValueError(
            f"No matches: control={len(src_idx)} target={len(tgt_idx)} in {pt_path}. "
            f"Check --control_regex / --target_regex against your crop paths."
        )
    return (Z[src_idx].astype(np.float32, copy=False),
            Z[tgt_idx].astype(np.float32, copy=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", required=True, help="latents .pt from export_latents_from_vae.py (train split)")
    ap.add_argument("--val_pt", required=True, help="latents .pt (val split)")
    ap.add_argument("--control_regex", required=True, help="regex matching control/source crop paths")
    ap.add_argument("--target_regex", required=True, help="regex matching perturbed/target crop paths")
    ap.add_argument("--out_dir", default="data/bbbc021")
    args = ap.parse_args()

    cre = re.compile(args.control_regex, re.IGNORECASE)
    tre = re.compile(args.target_regex, re.IGNORECASE)
    os.makedirs(args.out_dir, exist_ok=True)

    src_tr, tgt_tr = _split(args.train_pt, cre, tre)
    src_va, tgt_va = _split(args.val_pt, cre, tre)

    np.save(os.path.join(args.out_dir, "src_train.npy"), src_tr)
    np.save(os.path.join(args.out_dir, "tgt_train.npy"), tgt_tr)
    np.save(os.path.join(args.out_dir, "src_val.npy"), src_va)
    np.save(os.path.join(args.out_dir, "tgt_val.npy"), tgt_va)
    print(f"[OK] src_train {src_tr.shape} tgt_train {tgt_tr.shape} "
          f"src_val {src_va.shape} tgt_val {tgt_va.shape} -> {args.out_dir}")


if __name__ == "__main__":
    main()
