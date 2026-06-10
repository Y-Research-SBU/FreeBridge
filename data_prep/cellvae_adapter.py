import importlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch


def _import_obj(import_str: str):
    """
    import_str: "package.module:Name"
    Example: "train_vae_full:VAE"

    """
    if ":" not in import_str:
        raise ValueError(f"model_import must be like 'pkg.mod:Name', got {import_str}")
    mod_name, obj_name = import_str.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, obj_name):
        raise AttributeError(f"Module '{mod_name}' has no attribute '{obj_name}'")
    return getattr(mod, obj_name)


def _load_ckpt_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # pytorch-lightning: {"state_dict": {...}}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # our VAE trainer: {"model": {...}, "zdim": ..., "img_size": ...}
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    # raw state dict (only if all values are tensors)
    if isinstance(ckpt, dict) and len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise RuntimeError(
        f"Unsupported checkpoint format. "
        f"keys={list(ckpt.keys())[:20] if isinstance(ckpt, dict) else type(ckpt)}"
    )


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes=("model.", "vae.", "net.", "module.")):
    keys = list(state_dict.keys())
    for p in prefixes:
        if any(k.startswith(p) for k in keys):
            new_sd = { (k[len(p):] if k.startswith(p) else k): v for k, v in state_dict.items() }
            return new_sd, p
    return state_dict, None


def _guess_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class CellVAEAdapterConfig:
    ckpt_path: str
    model_import: str                     # e.g. "train_vae_full:VAE"
    model_kwargs_json: str = "{}"         # JSON string
    image_size: Tuple[int, int, int] = (3, 64, 64)  # (C,H,W)
    device: str = "cuda"
    strict: bool = True                  # load_state_dict strict
    # If your encoder returns (mu, logvar) you can select:
    encode_use: str = "mu"                # "mu" | "sample"
    # Expected decoder output range. If unknown, leave "auto".
    out_range: str = "auto"               # "auto" | "0_1" | "neg1_1"


class CellVAEAdapter(torch.nn.Module):
    """
   Common encode/decode interface.
      - decode(z) -> (B,C,H,W)
      - encode(x) -> z (B,D)  [optional but we implement best-effort]
    """
    def __init__(self, cfg: CellVAEAdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.device_ = _guess_device(cfg.device)

        obj = _import_obj(cfg.model_import)
        kwargs: Dict[str, Any] = json.loads(cfg.model_kwargs_json)

        model = obj(**kwargs) if callable(obj) else obj
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Imported object must produce nn.Module, got {type(model)}")

        sd = _load_ckpt_state_dict(cfg.ckpt_path)
        sd, stripped = _strip_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=cfg.strict)

        # Print key info only
        print(f"[CellVAEAdapter] Loaded ckpt: {cfg.ckpt_path}")
        if stripped:
            print(f"[CellVAEAdapter] Stripped prefix: '{stripped}'")
        print(f"[CellVAEAdapter] missing={len(missing)} unexpected={len(unexpected)} strict={cfg.strict}")
        if len(missing) > 0:
            print("  missing (first 20):", missing[:20])
        if len(unexpected) > 0:
            print("  unexpected (first 20):", unexpected[:20])

        self.model = model.eval().to(self.device_)
        self.image_size = tuple(int(x) for x in cfg.image_size)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,D)
        returns: (B,C,H,W)
        """
        z = z.to(self.device_)
        if hasattr(self.model, "decode") and callable(getattr(self.model, "decode")):
            x = self.model.decode(z)
        elif hasattr(self.model, "decoder") and callable(getattr(self.model, "decoder")):
            x = self.model.decoder(z)
        elif hasattr(self.model, "dec") and callable(getattr(self.model, "dec")):
            x = self.model.dec(z)
        else:
            raise AttributeError("Your VAE model has no .decode / .decoder / .dec method.")

        if not torch.is_tensor(x):
            raise TypeError(f"Decoder output must be Tensor, got {type(x)}")

        # allow flattened
        if x.dim() == 2:
            B = x.shape[0]
            C, H, W = self.image_size
            need = C * H * W
            if x.shape[1] != need:
                raise RuntimeError(f"Decoder returned (B,{x.shape[1]}) but C*H*W={need}. "
                                   f"Fix image_size or your decoder output.")
            x = x.view(B, C, H, W)

        if x.dim() != 4:
            raise RuntimeError(f"Decoder must return 4D (B,C,H,W). got {tuple(x.shape)}")

        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) -> z: (B,D)
        Best-effort: supports common APIs.
        """
        x = x.to(self.device_)
        if hasattr(self.model, "encode") and callable(getattr(self.model, "encode")):
            out = self.model.encode(x)
        elif hasattr(self.model, "encoder") and callable(getattr(self.model, "encoder")):
            out = self.model.encoder(x)
        elif hasattr(self.model, "enc") and callable(getattr(self.model, "enc")):
            out = self.model.enc(x)
        else:
            raise AttributeError("Your VAE model has no .encode / .encoder / .enc method.")

        # common cases:
        # - returns z
        # - returns (mu, logvar)
        # - returns (z, mu, logvar)
        if torch.is_tensor(out):
            z = out
        elif isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
            if len(out) == 2 and torch.is_tensor(out[1]):
                # (mu, logvar)  -- this is what train_vae_full.VAE.enc returns
                mu, logvar = out[0], out[1]
                z = (mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)) if self.cfg.encode_use == "sample" else mu
            elif len(out) >= 3 and torch.is_tensor(out[1]) and torch.is_tensor(out[2]):
                # (z, mu, logvar)
                z0, mu, logvar = out[0], out[1], out[2]
                if self.cfg.encode_use == "sample":
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                elif self.cfg.encode_use == "mu":
                    z = mu
                else:
                    z = z0
            else:
                z = out[0]
        elif isinstance(out, dict):
            # try keys
            for k in ["mu", "mean", "z", "latent"]:
                if k in out and torch.is_tensor(out[k]):
                    z = out[k]
                    break
            else:
                raise RuntimeError(f"Encoder dict has no known latent keys. keys={list(out.keys())}")
        else:
            raise RuntimeError(f"Unsupported encoder output type: {type(out)}")

        if z.dim() > 2:
            z = z.view(z.shape[0], -1)
        return z

    def to_01(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert decoded output to [0,1] for saving images.
        """
        x = x.detach()
        if self.cfg.out_range == "0_1":
            return x.clamp(0.0, 1.0)
        if self.cfg.out_range == "neg1_1":
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

        # auto
        mn = float(x.min().item())
        mx = float(x.max().item())
        if mn < -0.2 and mx <= 1.2:
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)