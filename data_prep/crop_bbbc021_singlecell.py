#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tifffile as tiff
except Exception as e:
    raise RuntimeError("Please install tifffile: pip install tifffile") from e

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("Please install pillow: pip install pillow") from e


# -------------------------
# utils
# -------------------------
W_PAT = re.compile(r"_w(\d)(?=[^0-9])", re.IGNORECASE)  # match "_w1" etc.

def robust_rescale_to_u8(x: np.ndarray, lo_q=1.0, hi_q=99.0) -> np.ndarray:
    xf = x.astype(np.float32)
    lo = np.percentile(xf, lo_q)
    hi = np.percentile(xf, hi_q)
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (xf - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)

def crop_square_with_padding(img: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    h, w, c = img.shape
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    out = np.zeros((size, size, c), dtype=img.dtype)

    sy0 = max(0, y0)
    sx0 = max(0, x0)
    sy1 = min(h, y1)
    sx1 = min(w, x1)

    dy0 = sy0 - y0
    dx0 = sx0 - x0
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)

    out[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
    return out


# -------------------------
# Channel discovery and selection
# -------------------------
def parse_priority(s: str) -> List[Tuple[int, int, int]]:
    """
    "124,123,234,134" -> [(1,2,4),(1,2,3),(2,3,4),(1,3,4)]
    """
    s = s.strip()
    if not s:
        raise ValueError("--priority cannot be empty")
    out: List[Tuple[int, int, int]] = []
    for token in s.split(","):
        token = token.strip()
        if len(token) != 3 or any(ch not in "12345" for ch in token):
            raise ValueError(f"Bad priority token: '{token}'. Use like 124,123,234")
        a, b, c = (int(token[0]), int(token[1]), int(token[2]))
        if len({a, b, c}) != 3:
            raise ValueError(f"Priority token must have 3 distinct channels: '{token}'")
        out.append((a, b, c))
    return out

def find_channel_files_any(raw_root: Path) -> Dict[str, Dict[int, Path]]:
    """
    Returns: base_key -> {w:int -> path}
    base_key = filename with suffix starting at '_wX' removed.
    """
    # case-insensitive .tif/.tiff matching (.TIF/.TIFF too)
    files = [f for f in raw_root.rglob("*") if f.suffix.lower() in (".tif", ".tiff")]
    m: Dict[str, Dict[int, Path]] = {}
    for p in files:
        name = p.name
        mm = W_PAT.search(name)
        if not mm:
            continue
        w = int(mm.group(1))
        # base_key: remove from '_wX' onward
        base = re.sub(r"_w\d.*$", "", name, flags=re.IGNORECASE)
        if not base:
            continue
        m.setdefault(base, {})[w] = p
    return m

def choose_combo(available_ws: List[int], priority: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
    S = set(available_ws)
    for combo in priority:
        if set(combo).issubset(S):
            return combo
    return None


# -------------------------
# mask path
# -------------------------
# -------------------------
# Mask lookup
# -------------------------
def _mask_matches_name(name: str, base_key: str) -> bool:
    """base_key must be followed by _w<digit>, a non-alphanumeric char, or end-of-name,
    so that s1 does NOT match s10."""
    name_l = name.lower()
    key = re.escape(base_key.lower())
    return re.search(rf"{key}(?:_w\d|[^a-z0-9]|$)", name_l) is not None


def guess_mask_path(mask_root: Path, base_key: str, any_channel_path: Path) -> Optional[Path]:
    """
    Robust matching.

    HF mask filename:
    BBBC021_v1_images_Week1_22123_Week1_150607_B02_s1_w1XXXX_mask.tif

    raw base_key:
    Week1_150607_B02_s1

    Therefore we must search by wildcard.
    """

    # case-insensitive scan: match any base_key + "mask" .tif/.tiff regardless of case
    hits = [
        p for p in mask_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and _mask_matches_name(p.name, base_key)
        and "mask" in p.name.lower()
    ]
    if hits:
        return sorted(hits)[0]

    # -------------------------
    # fallback: old behavior
    # -------------------------
    candidates: List[Path] = []

    for suffix in [".tif", ".tiff", ".TIF", ".TIFF"]:
        candidates += [
            mask_root / f"{base_key}_cp_masks{suffix}",
            mask_root / f"{base_key}_masks{suffix}",
            mask_root / f"{base_key}{suffix}",
        ]

    stem_any = any_channel_path.stem
    candidates += [
        mask_root / f"{stem_any}_cp_masks.tif",
        mask_root / f"{stem_any}_cp_masks.tiff",
        mask_root / f"{stem_any}.tif",
        mask_root / f"{stem_any}.tiff",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None



# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="Unzipped plate raw folder")
    ap.add_argument("--mask_root", type=str, required=True, help="Extracted masks folder for this week")
    ap.add_argument("--out_root", type=str, required=True, help="Output folder for crops")
    ap.add_argument("--crop_size", type=int, default=96)
    ap.add_argument("--padding", type=int, default=8, help="(kept for compatibility; not used except bbox center)")
    ap.add_argument("--min_area", type=int, default=120)

    # Channel selection priority
    ap.add_argument(
        "--priority",
        type=str,
        default="124,123,234,134",
        help="3-channel combos in priority order, e.g. '124,123,234,134'",
    )

    # keep old arg for backward compatibility, but we will interpret it differently:
    # now it indexes into the SELECTED 3 channels (0/1/2)
    ap.add_argument(
        "--rgb_idx",
        type=str,
        default="0,1,2",
        help="Indices into the SELECTED 3 channels for (R,G,B), e.g. 0,1,2",
    )

    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    mask_root = Path(args.mask_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    priority = parse_priority(args.priority)

    # parse rgb_idx but now must be 0..2 into selected-3
    parts = [int(x.strip()) for x in args.rgb_idx.split(",")]
    if len(parts) != 3 or any(p not in (0, 1, 2) for p in parts):
        raise ValueError("--rgb_idx must be like '0,1,2' with indices in {0,1,2}")
    if len(set(parts)) != 3:
        raise ValueError("--rgb_idx must contain three distinct indices, e.g. 0,1,2")
    r_i, g_i, b_i = parts

    # 1) collect all channels
    fields = find_channel_files_any(raw_root)
    if not fields:
        raise RuntimeError(f"No channel files found under: {raw_root}")

    # 2) for each field choose a 3-channel combo
    selected: List[Tuple[str, Tuple[int, int, int], Tuple[Path, Path, Path]]] = []
    for base_key, w2p in fields.items():
        combo = choose_combo(list(w2p.keys()), priority)
        if combo is None:
            continue
        pA, pB, pC = w2p[combo[0]], w2p[combo[1]], w2p[combo[2]]
        selected.append((base_key, combo, (pA, pB, pC)))

    if not selected:
        # Summary
        ws_seen = set()
        for _, w2p in fields.items():
            ws_seen |= set(w2p.keys())
        raise RuntimeError(
            f"No 3-channel fields matched priority under: {raw_root}. "
            f"Channels seen (union): {sorted(ws_seen)}. "
            f"Priority={priority}"
        )

    # write per-field selection metadata (useful later)
    meta_csv = out_root / "_field_channel_choice.csv"
    with open(meta_csv, "w", encoding="utf-8") as f:
        f.write("base_key,combo,paths\n")
        for base_key, combo, paths in selected:
            f.write(f"{base_key},{combo[0]}{combo[1]}{combo[2]},{paths[0].name}|{paths[1].name}|{paths[2].name}\n")

    n_fields = 0
    n_cells = 0
    n_skipped_no_mask = 0
    n_skipped_small = 0

    for base_key, combo, paths3 in selected:
        n_fields += 1

        # read the 3 chosen channels, in the combo order
        chans = []
        for p in paths3:
            arr = tiff.imread(str(p))
            if arr.ndim != 2:
                arr = np.squeeze(arr)
            chans.append(arr)

        # build RGB using indices into selected-3
        R = robust_rescale_to_u8(chans[r_i])
        G = robust_rescale_to_u8(chans[g_i])
        B = robust_rescale_to_u8(chans[b_i])
        rgb = np.stack([R, G, B], axis=-1)

        # locate mask (use first chosen channel as reference for naming)
        mask_path = guess_mask_path(mask_root, base_key, paths3[0])
        if mask_path is None:
            n_skipped_no_mask += 1
            continue

        mask = tiff.imread(str(mask_path))
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            mask = mask[..., 0]
        if mask.shape != rgb.shape[:2]:
            raise RuntimeError(
                f"Mask/raw shape mismatch for {base_key}: mask={mask.shape}, "
                f"raw={rgb.shape[:2]}, mask_path={mask_path}"
            )

        labels = np.unique(mask)
        labels = labels[labels != 0]
        if labels.size == 0:
            continue

        # per label crop
        combo_str = f"{combo[0]}{combo[1]}{combo[2]}"
        for lab in labels:
            ys, xs = np.where(mask == lab)
            area = ys.size
            if area < args.min_area:
                n_skipped_small += 1
                continue

            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2

            patch = crop_square_with_padding(rgb, cy, cx, args.crop_size)

            # save; include combo in filename so you can group later if needed
            out_path = out_root / f"{base_key}_c{combo_str}_cell{int(lab):04d}.png"
            Image.fromarray(patch).save(out_path, optimize=True)
            n_cells += 1

    print(
        f"[OK] fields_selected={n_fields}, cells_saved={n_cells}, "
        f"skipped_no_mask={n_skipped_no_mask}, skipped_small={n_skipped_small}, "
        f"meta_csv={meta_csv}"
    )


if __name__ == "__main__":
    main()
