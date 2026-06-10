from typing import Any, List
import os
import copy
import math
import gc
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from rich.console import Console

from .network import build_net
from .state_cost import build_state_cost
from .evaluator import build_evaluator
from .sde import build_basedrift, sdeint

from .dataset import PairDataset, SplineDataset, SplineIWDataset
from . import gaussian_path as gpath_lib
from . import path_integral as pi_lib
from . import match_loss as match_lib
from . import utils

console = Console()


class FreeBridgeLitModule(pl.LightningModule):
    """Lightning module for FreeBridge training."""

    def __init__(self, cfg, p0, p1, p0_val, p1_val):
        super().__init__()

        os.makedirs("figs", exist_ok=True)

        self.cfg = cfg
        self.p0 = p0
        self.p1 = p1
        self.p0_val = p0_val
        self.p1_val = p1_val

        # Problem
        self.sigma = cfg.prob.sigma
        self.V = build_state_cost(cfg)
        self.basedrift = build_basedrift(cfg)
        self.evaluator = build_evaluator(cfg)

        # SB model
        self.direction = None
        self.fwd_net = build_net(cfg)
        self.bwd_net = build_net(cfg)

        # Ensure train_data exists before Lightning requests train_dataloader.
        # It will be overwritten in validation_epoch_end() after the first R-step.
        self.train_data = None

    # -------------------------
    # Convenience / config
    # -------------------------
    def print(self, content: str, prefix: bool = True):
        if getattr(self, "trainer", None) is None:
            return
        if self.trainer.is_global_zero:
            if prefix:
                now = f"[[cyan]{datetime.now():%Y-%m-%d %H:%M:%S}[/cyan]]"
                if self.direction is None:
                    base = f"[[blue]Init[/blue]] "
                else:
                    base = f"[[blue]Ep {self.current_epoch} ({self.direction})[/blue]] "
                console.print(now, highlight=False, end="")
                console.print(base, end="")
            console.print(f"{content}")

    @property
    def wandb_logger(self):
        return self.loggers[-1]

    @property
    def is_img_prob(self):
        return self.cfg.prob.name in ["afhq"]

    @property
    def logging_batch_idxs(self):
        if getattr(self, "trainer", None) is None:
            return np.array([], dtype=int)
        return np.linspace(0, self.trainer.num_training_batches - 1, 10).astype(int)

    @property
    def ocfg(self):
        return self.cfg.optim

    @property
    def ccfg(self):
        return self.cfg.csoc

    @property
    def mcfg(self):
        return self.cfg.matching

    @property
    def device(self):
        return self.fwd_net.parameters().__next__().device

    @property
    def net(self):
        d = self.direction or "fwd"
        return self.fwd_net if d == "fwd" else self.bwd_net

    @property
    def direction_r(self):
        return "bwd" if self.direction == "fwd" else "fwd"

    # -------------------------
    # Drift / sampling utilities
    # -------------------------
    def build_ft(self, direction: str):
        def ft(x, t):
            """
            x: (B, D)
            t: (B,)
            out: (B, D)
            """
            B, D = x.shape
            sign = 1.0 if direction == "fwd" else -1.0
            assert t.shape == (B,) and torch.allclose(t, t[0] * torch.ones_like(t))
            return sign * self.basedrift(x.unsqueeze(1), t[0].reshape(1)).squeeze(1)

        return ft

    def build_ut(self, direction: str, backprop_snet: bool = False):
        """
        ut: (x: (B, D), t: (B,)) -> (B, D)
        """
        net = self.fwd_net if direction == "fwd" else self.bwd_net
        if self.cfg.field == "vector":
            ut = net
        elif self.cfg.field == "potential":

            def ut(x, t):
                with torch.enable_grad():
                    x = x.detach().clone()
                    x.requires_grad_(True)
                    out = net(x, t)
                    return torch.autograd.grad(out.sum(), x, create_graph=backprop_snet)[0]

        else:
            raise ValueError(f"Unsupported field: {self.cfg.field}!")
        return ut

    def build_drift(self, direction: str, backprop_snet: bool = False):
        ft = self.build_ft(direction)
        ut = self.build_ut(direction, backprop_snet=backprop_snet)
        return lambda x, t: ut(x, t) + ft(x, t)

    @torch.no_grad()
    def sample(self, xinit, log_steps: int, direction: str, drift=None, nfe=None, verbose: bool = False):
        drift = self.build_drift(direction) if drift is None else drift
        diffusion = lambda x, t: self.sigma
        nfe = nfe or self.cfg.nfe
        return sdeint(
            xinit,
            drift,
            diffusion,
            direction,
            nfe=nfe,
            log_steps=log_steps,
            verbose=verbose,
        )

    def sample_t(self, batch: int):
        if self.mcfg.loss == "eam":
            t0 = torch.rand(1)
            t = (t0 + math.sqrt(2) * torch.arange(batch)) % 1
            t.clamp_(min=0.001, max=0.999)
        elif self.mcfg.loss == "bm":
            eps = 1e-4
            t = torch.rand(batch).reshape(-1) * (1 - 2 * eps) + eps
        else:
            raise ValueError(f"Unsupported matching loss: {self.mcfg.loss}!")
        assert t.shape == (batch,)
        return t

    def sample_gpath(self, batch):
        gpath = gpath_lib.EndPointGaussianPath(
            batch["mean_t"][0],
            batch["mean_xt"],
            batch["gamma_s"][0],
            batch["gamma_xs"],
            self.sigma,
            self.basedrift,
        )
        x0, x1 = batch["x0"], batch["x1"]
        B, D = x0.shape

        T = B if self.mcfg.loss == "bm" else self.mcfg.batch_t
        if not self.ccfg.IW:
            t = self.sample_t(T).to(x0)
            with torch.no_grad():
                xt = gpath.sample_xt(t, N=1)
        else:
            IW_t, IW_xs, weights = batch["IW_t"][0], batch["IW_xs"], batch["weights"]
            assert (weights > 0).all()
            assert torch.allclose(weights.sum(dim=1), torch.ones(B).to(weights))

            rand_idx = torch.randint(low=0, high=len(IW_t), size=(T,))
            t = IW_t[rand_idx]
            xt = pi_lib.impt_weighted(t, IW_xs, weights).unsqueeze(1)
        assert t.shape == (T,) and xt.shape == (B, 1, T, D)

        if self.mcfg.loss == "bm":
            assert B == T
            vt = gpath.ut(t, xt, self.direction or "fwd")
            xt = xt[torch.arange(B), 0, torch.arange(B)]
            vt = vt[torch.arange(B), 0, torch.arange(B)]
            assert xt.shape == vt.shape == (B, D)
        else:
            vt = None
            xt = xt.squeeze(1)
            assert xt.shape == (B, T, D)

        return x0, x1, t, xt, vt

    # -------------------------
    # Data initialization
    # -------------------------
    def localize(self, p, stream: int = 0):
        g = torch.Generator()
        rank = int(getattr(self, "global_rank", 0))
        epoch = int(getattr(self, "current_epoch", 0))
        base_seed = int(getattr(self.cfg, "seed", 0) or 0)
        # deterministic across runs given cfg.seed, while still varying per epoch/rank/stream
        g.manual_seed(base_seed + 1000003 * epoch + 1009 * rank + stream)
        local_p = copy.copy(p)   # shallow: do not duplicate the latent array
        local_p.set_generator(g)
        return local_p

    def _init_train_data_if_needed(self):
        """
        Build a placeholder SplineDataset so Lightning can start training before the first R-step.
        After the first validation_epoch_end(), train_data will be replaced by fitted Gaussian paths.
        """
        if self.train_data is not None:
            return

        ccfg = self.ccfg
        T, S = ccfg.T_mean, ccfg.T_gamma

        totalB, n_device = ccfg.B, utils.n_device()
        B = totalB // max(1, n_device)
        if B <= 0:
            raise ValueError(f"per-device batch B={B} <= 0; csoc.B too small for n_device")

        self.print(f"[Data] Init placeholder train_data (straight-line) with B={B} T={T} S={S}.")

        x0 = self.localize(self.p0, stream=0)(B).detach().cpu()
        x1 = self.localize(self.p1, stream=1)(B).detach().cpu()
        assert x0.shape == x1.shape and x0.ndim == 2

        mean_t = torch.linspace(0, 1, T)
        gamma_s = torch.linspace(0, 1, S)

        mean_xt = (1 - mean_t[None, :, None]) * x0[:, None, :] + mean_t[None, :, None] * x1[:, None, :]
        gamma_xs = torch.zeros(B, S, 1)

        self.train_data = SplineDataset(mean_t, mean_xt, gamma_s, gamma_xs, expand_factor=ccfg.epd_fct)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self._init_train_data_if_needed()

    # -------------------------
    # Lightning hooks
    # -------------------------
    def train_dataloader(self):
        self._init_train_data_if_needed()
        return DataLoader(
            self.train_data,
            num_workers=self.ocfg.num_workers,
            batch_size=self.ocfg.batch_size,
            persistent_workers=self.ocfg.num_workers > 0,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        totalB, n_device = self.ccfg.B, utils.n_device()
        B = totalB // max(1, n_device)
        if B <= 0:
            raise ValueError(f"per-device batch B={B} <= 0; csoc.B too small for n_device")
        self.print(f"[Data] Building {totalB} train_data ...")
        self.print(f"[Data] Found {n_device} devices, each will generate {B} samples ...")

        x0 = self.localize(self.p0, stream=0)(B)
        x1 = self.localize(self.p1, stream=1)(B)
        return DataLoader(
            PairDataset(x0, x1),
            num_workers=self.ocfg.num_workers,
            batch_size=self.ccfg.mB,
            persistent_workers=self.ocfg.num_workers > 0,
            shuffle=False,
            pin_memory=True,
        )

    def training_step(self, batch: Any, batch_idx: int):
        x0, x1, t, xt, vt = self.sample_gpath(batch)
        direction = self.direction or "fwd"

        if self.mcfg.loss == "bm":
            ut = self.build_ut(direction, backprop_snet=True)
            loss = match_lib.bm_loss(ut, xt, t, vt)
        elif self.mcfg.loss == "eam":
            loss = match_lib.eam_loss_trajs(
                self.net,
                xt,
                t,
                x0,
                x1,
                self.sigma,
                self.direction,
                lap=self.mcfg.lap,
            )
        else:
            raise ValueError(f"Unsupported match_loss: {self.mcfg.loss}!")

        if torch.isfinite(loss):
            self.log("train/loss", loss, on_step=True, on_epoch=True)
        else:
            self.print(f"Skipping iteration because loss is {loss.item()}.")
            return None

        if batch_idx in self.logging_batch_idxs:
            self.print(f"[M-step] batch idx: {batch_idx+1}/{self.trainer.num_training_batches} ...")

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        self.direction = self.direction_r
        self.print("", prefix=False)

    def compute_coupling(self, batch, direction, eval_coupling: bool):
        x0, x1, T = batch["x0"], batch["x1"], self.ccfg.T_mean
        if direction is None:
            t = torch.linspace(0, 1, T).to(x0)
            xt = (1 - t[None, :, None]) * x0.unsqueeze(1) + t[None, :, None] * x1.unsqueeze(1)
        else:
            xinit = x0 if direction == "fwd" else x1
            output = self.sample(xinit, log_steps=T, direction=direction)
            t, xt = output["t"], output["xs"]

            # For bbbc021, xs[:,-1] is the generated target only in the forward
            # direction. In the backward direction xs[:,-1] is the *input* target,
            # so evaluating there would compare the reference against itself and
            # report misleadingly good metrics. Skip eval on backward.
            skip_eval = (str(self.cfg.prob.name) == "bbbc021" and direction == "bwd")
            if eval_coupling and not skip_eval:
                metrics = self.evaluator(output)
                for k, v in metrics.items():
                    if isinstance(v, (int, float, np.number, torch.Tensor)):
                        self.log(f"metrics/{k}", v, on_epoch=True)
                    else:
                        self.print(f"[Eval] {k}: {v}")

        return t, xt

    def validation_step(self, batch: Any, batch_idx: int):
        log_step = batch_idx == 0
        ccfg, direction = self.ccfg, self.direction
        postfix = f"{self.current_epoch:03d}" if direction is not None else "init"

        (B, D), T, S, sigma = batch["x0"].shape, ccfg.T_mean, ccfg.T_gamma, self.sigma

        eval_coupling = log_step and self.cfg.eval_coupling
        self.print(f"[R-step] Simulating {direction or 'init'} coupling ...")
        t, xt = self.compute_coupling(batch, direction, eval_coupling)
        self.print(f"[R-step] Simulated xt.shape={tuple(xt.shape)}!")

        assert xt.shape == (B, T, D) and t.shape == (T,)

        s = torch.linspace(0, 1, S).to(t)
        ys = torch.zeros(B, S, 1).to(xt)

        gpath = gpath_lib.EndPointGaussianPath(t, xt, s, ys, sigma, self.basedrift)

        # FreeBridge: attach the REAL minibatch empirical support bank
        # (control + perturbed endpoints) so the support state cost matches the
        # paper definition even after the gpath endpoints become model-generated.
        if str(self.cfg.prob.name) == "bbbc021":
            support_bank = torch.cat([batch["x0"], batch["x1"]], dim=0).detach()
            gpath.register_buffer("support_bank", support_bank)

        if self.is_img_prob:
            loss_fn = gpath_lib.build_img_loss_fn(gpath, sigma, self.V, ccfg)
        else:
            loss_fn = gpath_lib.build_loss_fn(gpath, sigma, self.V, ccfg)

        with torch.enable_grad():
            verbose = log_step and self.trainer.is_global_zero
            result = gpath_lib.fit(ccfg, gpath, direction or "fwd", loss_fn, verbose=verbose)

        self.print(f"[R-step] Fit {B} gaussian paths!")

        xt = gpath.mean.xt.detach().clone()
        ys = gpath.gamma.xt.detach().clone()
        assert xt.shape == (B, T, D) and ys.shape == (B, S, 1)

        output = {"mean_t": t, "mean_xt": xt, "gamma_s": s, "gamma_xs": ys}

        if ccfg.IW:
            with torch.no_grad():
                iw_output = pi_lib.impt_sample_xs(ccfg, gpath, sigma, direction or "fwd", V=self.V)
            output.update(iw_output)
            self.print(f"[R-step] Compute IW weights shape={tuple(iw_output['weights'].shape)}!")

        if ccfg.name == "opinion":
            tt = torch.linspace(0, 1, self.cfg.pdrift.S).to(t)
            mf_x = gpath.sample_xt(tt, N=1).squeeze(1)
            assert mf_x.shape == (B, len(tt), D)
            output["mf_x"] = mf_x.detach().cpu()

        return output

    def validation_epoch_end(self, outputs: List[Any]):
        if self.cfg.prob.name == "opinion":
            mf_xs = utils.gather(outputs, "mf_x")
            self.basedrift.set_mf_drift(mf_xs)
            self.print(f"[Opinion] Set MF drift shape={tuple(mf_xs.shape)}!")

        ccfg = self.ccfg
        T, S, D = ccfg.T_mean, ccfg.T_gamma, self.cfg.dim

        mean_t = outputs[0]["mean_t"].detach().cpu()
        gamma_s = outputs[0]["gamma_s"].detach().cpu()
        assert mean_t.shape == (T,) and gamma_s.shape == (S,)

        mean_xt = utils.gather(outputs, "mean_xt")
        gamma_xs = utils.gather(outputs, "gamma_xs")
        B = mean_xt.shape[0]
        assert mean_xt.shape == (B, T, D)
        assert gamma_xs.shape == (B, S, 1)

        self.train_data = SplineDataset(mean_t, mean_xt, gamma_s, gamma_xs, expand_factor=ccfg.epd_fct)
        self.print(f"[Data] Fit total {B} gaussian paths as train_data!")

        if ccfg.IW:
            iN, iS = ccfg.IW_N, ccfg.IW_S
            IW_t = outputs[0]["IW_t"].detach().cpu()
            IW_xs = utils.gather(outputs, "IW_xs")
            weights = utils.gather(outputs, "weights")
            assert IW_t.shape == (iS,) and IW_xs.shape == (B, iN, iS, D)
            assert weights.shape == (B, iN)
            self.print(f"[Data] Computed importance weights shape={tuple(weights.shape)} as train_data!")
            self.train_data = SplineIWDataset(self.train_data, IW_t, IW_xs, weights)

        if self.direction is None:
            self.direction = "fwd"
            self.print("", prefix=False)

        torch.cuda.empty_cache()
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.ocfg.lr,
            weight_decay=self.ocfg.wd,
            eps=self.ocfg.eps,
        )

        if self.ocfg.get("scheduler", "cosine") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.ocfg.num_iterations,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.net.update_ema()