import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .export_latents_mu import load_vae_from_ckpt
except ImportError:
    from export_latents_mu import load_vae_from_ckpt

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _is_image_path(p: str) -> bool:
    return Path(p).suffix.lower() in IMG_EXTS


def _resolve_path(p: str, crops_root: Optional[str]) -> str:
    p = p.strip()
    if os.path.isabs(p):
        return p
    if crops_root is None:
        return p  # keep relative; existence check later
    # avoid double-prefixing if the relative path already starts with crops_root
    if os.path.exists(p):
        return p
    if crops_root is not None:
        candidate = os.path.join(crops_root, p)
        if os.path.exists(candidate):
            return candidate
    return p


def _load_list_paths(list_file: str, crops_root: Optional[str]) -> List[str]:
    with open(list_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    out: List[str] = []
    skipped_ext = 0
    skipped_missing = 0

    for ln in lines:
        p = _resolve_path(ln, crops_root)

        # skip obvious non-images (zip, txt, etc.)
        if not _is_image_path(p):
            skipped_ext += 1
            continue

        # skip missing
        if not os.path.exists(p):
            skipped_missing += 1
            continue

        out.append(p)

    print(f"[list] total lines={len(lines)} kept_images={len(out)} skipped_nonimg={skipped_ext} skipped_missing={skipped_missing}")
    if len(out) == 0:
        raise RuntimeError("No valid image paths after filtering. raw_list.txt likely points to wrong root.")
    return out


class PathImageDataset(Dataset):
    def __init__(self, paths: List[str], size: int):
        self.paths = paths
        self.size = int(size)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), resample=Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return x, p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list_file", required=True)
    ap.add_argument("--crops_root", default=None, help="used only if list_file has relative paths")
    ap.add_argument("--out_pt", required=True)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--sample", action="store_true",
                    help="Export stochastic VAE samples instead of deterministic mu (default: mu).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck, vae, zdim, img_size = load_vae_from_ckpt(args.ckpt, device=device)
    vae.eval()

    paths = _load_list_paths(args.list_file, args.crops_root)
    if args.limit is not None:
        paths = paths[: int(args.limit)]

    ds = PathImageDataset(paths, size=img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    zs = []
    out_paths = []

    for xb, ps in dl:
        xb = xb.to(device, non_blocking=True)
        mu, logvar = vae.enc(xb)

        if args.sample:
            eps = torch.randn_like(mu)
            out = mu + eps * torch.exp(0.5 * logvar)
        else:
            out = mu

        zs.append(out.detach().cpu())
        out_paths.extend(list(ps))

    Z = torch.cat(zs, dim=0)
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Z": Z, "paths": out_paths, "zdim": int(zdim), "img_size": int(img_size)}, str(out_pt))
    print(f"[OK] saved {tuple(Z.shape)} -> {out_pt}")


if __name__ == "__main__":
    main()