import os
import glob
import math
import argparse
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def read_crop_rgb(path: str, img_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), resample=Image.BICUBIC)
    x = (np.asarray(img, dtype=np.float32) / 255.0)
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    return x

class CropDataset(Dataset):
    def __init__(self, root: str, img_size: int = 96, limit: Optional[int] = None):
        exts = ["png", "jpg", "jpeg", "tif", "tiff", "PNG", "JPG", "JPEG", "TIF", "TIFF"]
        paths: List[str] = []
        for e in exts:
            paths += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
        paths = sorted(paths)
        if limit is not None:
            paths = paths[: int(limit)]
        if len(paths) == 0:
            raise RuntimeError(f"No images under {root}")
        self.paths = paths
        self.img_size = int(img_size)

    def __len__(self) -> int: return len(self.paths)
    def __getitem__(self, i: int) -> torch.Tensor: return read_crop_rgb(self.paths[i], self.img_size)

class Enc(nn.Module):
    def __init__(self, net: nn.Sequential, fc_mu: nn.Linear, fc_logvar: nn.Linear):
        super().__init__()
        self.net = net
        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Dec(nn.Module):
    def __init__(self, fc: nn.Linear, net: nn.Sequential, in_hw: int, in_ch: int):
        super().__init__()
        self.fc = fc
        self.net = net
        self.in_hw = in_hw
        self.in_ch = in_ch

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.in_ch, self.in_hw, self.in_hw)
        return self.net(h)

class VAE(nn.Module):
    def __init__(self, enc: Enc, dec: Dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

def _sorted_layer_ids(sd: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    ids = set()
    for k in sd.keys():
        if k.startswith(prefix) and k.endswith(".weight"):
            mid = k[len(prefix):].split(".")[0]
            if mid.isdigit(): ids.add(int(mid))
    return sorted(ids)

def _infer_stride_padding(k: int) -> Tuple[int, int]:
    if k == 4: return 2, 1
    if k == 3: return 1, 1
    return 1, k // 2

def _build_vae_from_sd(sd: Dict[str, torch.Tensor], zdim: int) -> VAE:
    enc_ids = _sorted_layer_ids(sd, "enc.net.")
    dec_ids = _sorted_layer_ids(sd, "dec.net.")
    
    enc_layers: List[nn.Module] = []
    for lid in enc_ids:
        w = sd[f"enc.net.{lid}.weight"]
        b = sd.get(f"enc.net.{lid}.bias", None)
        out_ch, in_ch, k, _ = w.shape
        stride, pad = _infer_stride_padding(int(k))
        enc_layers.append(nn.Conv2d(in_ch, out_ch, int(k), stride=stride, padding=pad, bias=(b is not None)))
        enc_layers.append(nn.ReLU(inplace=True))
    enc_net = nn.Sequential(*enc_layers)

    in_features = sd["enc.fc_mu.weight"].shape[1]
    fc_mu = nn.Linear(in_features, zdim)
    fc_logvar = nn.Linear(in_features, zdim)

    dec_fc_out = sd["dec.fc.weight"].shape[0]
    first_w = sd[f"dec.net.{dec_ids[0]}.weight"]
    first_b = sd.get(f"dec.net.{dec_ids[0]}.bias", None)
    
    if first_b.numel() == first_w.shape[0]: first_in_ch = int(first_w.shape[1])
    else: first_in_ch = int(first_w.shape[0])

    dec_hw = int(round(math.sqrt(float(dec_fc_out) / float(first_in_ch))))
    dec_fc = nn.Linear(zdim, dec_fc_out)

    dec_layers: List[nn.Module] = []
    for lid in dec_ids:
        w = sd[f"dec.net.{lid}.weight"]
        b = sd.get(f"dec.net.{lid}.bias", None)
        k = int(w.shape[2])
        stride, pad = _infer_stride_padding(k)
        is_conv2d = (b.numel() == w.shape[0])
        
        if is_conv2d:
            out_ch, in_ch = int(w.shape[0]), int(w.shape[1])
            dec_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=pad))
        else:
            in_ch, out_ch = int(w.shape[0]), int(w.shape[1])
            dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad))
        
        if lid != dec_ids[-1]: dec_layers.append(nn.ReLU(inplace=True))
        elif out_ch == 3: dec_layers.append(nn.Sigmoid())

    return VAE(Enc(enc_net, fc_mu, fc_logvar), Dec(dec_fc, nn.Sequential(*dec_layers), dec_hw, first_in_ch))

def load_vae_from_ckpt(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"]
    zdim = int(ck.get("zdim", 64))
    vae = _build_vae_from_sd(sd, zdim=zdim).to(device)
    vae.load_state_dict(sd, strict=True)
    return ck, vae, zdim, int(ck.get("img_size", 96))