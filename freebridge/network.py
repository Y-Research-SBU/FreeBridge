import torch
import torch.nn as nn

from .ema import EMA
from .nn import (
    timestep_embedding,
    Unbatch,
    SiLU,
    ResNet_FC,
)


def build_net(cfg):
    """
    Builds the score / vector-field network wrapped with EMA + Unbatch.

    Supported combinations for cfg.net / cfg.field (non-UNet branch):
      - toy-potential
      - toy-vector
      - opinion-vector
      - mlp-vector   (added for generic latent vectors, e.g. dim=64)
    """
    if hasattr(cfg, "unet"):
        field = UNetVectorField(cfg.unet)
    else:
        key = f"{cfg.net}-{cfg.field}"
        table = {
            "toy-potential": ToyPotentialField,
            "toy-vector": ToyVectorField,
            "opinion-vector": OpinionVectorField,
            "mlp-vector": MLPVectorField,
        }
        ctor = table.get(key, None)
        if ctor is None:
            raise KeyError(
                f"Unknown net/field combo: '{key}'. "
                f"Valid keys: {sorted(table.keys())}. "
                f"(Hint: for BBBC021 latent dim=64, use net=mlp field=vector.)"
            )
        field = ctor(cfg.dim)

    return EMA(Unbatch(field), cfg.optim.ema_decay)


class ToyPotentialField(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128):
        super(ToyPotentialField, self).__init__()

        self.xt_module = ResNet_FC(data_dim + 1, hidden_dim, num_res_blocks=3)

        self.out_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,)
        """
        h = torch.hstack([t.reshape(-1, 1), x])
        h = self.xt_module(h)
        out = self.out_module(h)
        return out


class ToyVectorField(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 128,
        step_scale: int = 1000,
    ):
        super(ToyVectorField, self).__init__()

        self.step_scale = step_scale
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = nn.Sequential(
            nn.Linear(data_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,)
        """
        steps = t * self.step_scale
        t_emb = timestep_embedding(steps, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out + t_out)
        return out


class OpinionVectorField(nn.Module):
    def __init__(
        self, data_dim=1000, hidden_dim=256, time_embed_dim=128, step_scale=1000
    ):
        super(OpinionVectorField, self).__init__()

        self.step_scale = step_scale
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(data_dim, hid, num_res_blocks=5)

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,)
        """
        t = t * self.step_scale
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out + t_out)
        return out


class MLPVectorField(nn.Module):
    """
    Generic time-conditioned MLP vector field for latent vectors.

    Use with:
      net: mlp
      field: vector
      dim: <latent_dim>   (e.g., 64)
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        step_scale: int = 1000,
        n_layers: int = 3,
    ):
        super().__init__()
        self.step_scale = step_scale
        self.time_embed_dim = time_embed_dim

        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = [nn.Linear(data_dim, hidden_dim), SiLU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), SiLU()]
        self.x_module = nn.Sequential(*layers)

        self.out_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,)
        """
        steps = t * self.step_scale
        t_emb = timestep_embedding(steps, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        return self.out_module(x_out + t_out)


class UNetVectorField(nn.Module):
    def __init__(self, cfg, timesteps=1000):
        super(UNetVectorField, self).__init__()

        from .unet import UNetModel

        self.net = UNetModel(**cfg)
        self.timesteps = timesteps

    def forward(self, x, t) -> torch.Tensor:
        """
        x: (B, D) range: [-1,1]
        t: (B,)   timesteps (in [0,1] typically, then scaled)
        """
        B, D = x.shape
        assert t.shape == (B,)
        assert D == 3 * 64 * 64

        batch = {}
        batch["noisy_x"] = x.reshape(B, 3, 64, 64)
        timestep = t * self.timesteps

        return self.net(batch, timestep).reshape(B, D)