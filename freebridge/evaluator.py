from __future__ import annotations
import os
from pathlib import Path
import math
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as td


from .state_cost import build_obstacle_cost, congestion_cost, zero_cost_fn
from .utils import get_repo_path



# =========================
# Evaluator factory
# =========================

def build_evaluator(cfg):
    # bbbc021: compute latent-FID (+ optional MOA)
    if cfg.prob.name == "bbbc021":
        return BBBC021Evaluator(cfg)

    # tasks that don't use metrics here
    if cfg.prob.name in ["opinion", "afhq", "lidar"]:
        return DumpEvaluator()

    # default
    return CrowdNavEvaluator(cfg)


# =========================
# Small utils
# =========================

def cpu_everything(*args):
    return [a.cpu() for a in args] if len(args) > 1 else args.cpu()

def shuffle(t):
    """
    t: (B, *) --> (B, *)
    """
    return t[torch.randperm(t.shape[0])]

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)

def _cov(x):
    # x: (N, D)
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(1, x.shape[0])
    return mu, cov

def _sqrtm_psd(A, eps=1e-10):
    # symmetric PSD matrix sqrt via eigen decomposition
    A = (A + A.T) * 0.5
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T

def frechet_distance(x, y, eps=1e-6):
    """
    Fréchet distance between two Gaussian fits of x and y.
    x,y: (N,D)
    """
    mu1, C1 = _cov(x)
    mu2, C2 = _cov(y)
    diff = mu1 - mu2

    # Stabilize covariances
    D = C1.shape[0]
    C1 = C1 + np.eye(D) * eps
    C2 = C2 + np.eye(D) * eps

    # Compute trace term using PSD sqrt
    sqrtC1 = _sqrtm_psd(C1)
    M = sqrtC1 @ C2 @ sqrtC1
    sqrtM = _sqrtm_psd(M)

    tr = np.trace(C1) + np.trace(C2) - 2.0 * np.trace(sqrtM)
    return float(diff @ diff + tr)

def moa_knn_predict(x_gen, x_ref, y_ref):
    """
    1-NN predict labels of generated samples using reference set.
    Returns predicted labels array of shape (N,).
    """
    x_gen = np.asarray(x_gen, dtype=np.float32)
    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_ref = np.asarray(y_ref)

    # squared distances: ||a||^2 + ||b||^2 - 2 a.b
    aa = (x_gen * x_gen).sum(1, keepdims=True)      # (N,1)
    bb = (x_ref * x_ref).sum(1, keepdims=True).T    # (1,M)
    d2 = aa + bb - 2.0 * (x_gen @ x_ref.T)          # (N,M)
    nn = np.argmin(d2, axis=1)
    return y_ref[nn]


# =========================
# MMD loss (kept for CrowdNav)
# =========================

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


@torch.no_grad()
def est_entropy_cost(xt, std=0.2):
    """
    xt: (B, T, D) --> (B, T)
    """
    B, T, D = xt.shape

    normals = td.Normal(
        xt.reshape(B * T, D),
        std * torch.ones(B * T, D).to(xt),
    )
    indep_normals = td.Independent(normals, 1)

    xxt = xt.unsqueeze(1).expand(-1, B, -1, -1)
    assert xxt.shape == (B, B, T, D)

    log_pt_01 = indep_normals.log_prob(xxt.reshape(B, B * T, D)).reshape(B, B, T)
    pt = log_pt_01.exp().mean(dim=1)  # (B, T)

    log_pt = pt.log()
    assert not torch.isnan(log_pt).any()
    assert log_pt.shape == (B, T)
    return log_pt


# =========================
# Evaluators
# =========================

class DumpEvaluator:
    def __call__(self, samples):
        return {}


from .dataset import _resolve_data_path


class BBBC021Evaluator:
    """
    Latent-FID evaluator for BBBC021.

    Expects samples as dict with "xs": (B,T,D).
    Uses x1 = xs[:,-1] as generated target latents.
    Reference target latents loaded from cfg.data.tgt_val (fallback: tgt_train).

    Optional MOA:
      - if cfg.data.moa_val exists and is a .npy array aligned with tgt_val,
        will output simple predicted MOA distribution summary using 1-NN in latent space.
    """
    def __init__(self, cfg):
        self.cfg = cfg

        if hasattr(cfg, "data") and hasattr(cfg.data, "tgt_val"):
            tgt_path = cfg.data.tgt_val
        else:
            tgt_path = cfg.data.tgt_train

        self.x1_ref = np.load(_resolve_data_path(tgt_path))
        if self.x1_ref.ndim != 2:
            self.x1_ref = self.x1_ref.reshape(self.x1_ref.shape[0], -1)

        self.moa_ref = None
        if hasattr(cfg.data, "moa_val"):
            self.moa_ref = np.load(_resolve_data_path(cfg.data.moa_val))
            if len(self.moa_ref) != self.x1_ref.shape[0]:
                raise ValueError(f"moa_val length {len(self.moa_ref)} != tgt_val {self.x1_ref.shape[0]}")

        # cap reference size for speed
        self.max_ref = int(getattr(cfg, "fid_max_ref", 50000))

    @torch.no_grad()
    def __call__(self, samples):
        metrics = {}

        if not isinstance(samples, dict) or "xs" not in samples:
            return metrics

        xs = samples["xs"]
        if xs.ndim != 3:
            return metrics

        x1 = xs[:, -1]  # (B,D)
        x1_gen = _to_numpy(x1)
        if x1_gen.ndim != 2:
            x1_gen = x1_gen.reshape(x1_gen.shape[0], -1)

        ref = self.x1_ref
        if ref.shape[0] > self.max_ref:
            idx = np.random.default_rng(int(getattr(self.cfg, 'seed', 0) or 0)).choice(ref.shape[0], self.max_ref, replace=False)
            ref = ref[idx]
            moa_ref = self.moa_ref[idx] if self.moa_ref is not None else None
        else:
            moa_ref = self.moa_ref

        metrics["FID_latent"] = frechet_distance(x1_gen, ref)

        if moa_ref is not None:
            pred = moa_knn_predict(x1_gen, ref, moa_ref)
            uniq, cnt = np.unique(pred, return_counts=True)
            metrics["MOA_pred_unique"] = float(len(uniq))
            # NOTE: keep only numeric values in `metrics` (Lightning self.log cannot log strings).
            # The top-1 label is printed below, not logged.
            metrics["_text_MOA_pred_top1"] = str(uniq[np.argmax(cnt)])
        return metrics


class CrowdNavEvaluator:
    B = 1000
    D = 2

    def __init__(self, cfg) -> None:
        self.ccfg = cfg.csoc
        self.scfg = cfg.state_cost
        self.sigma = cfg.prob.sigma

        self.obstacle_cost = build_obstacle_cost(cfg.prob.name)
        self.sinkhorn_cfg = {"p": 2, "blur": 0.05, "scaling": 0.95}
        self.ref_x0, self.ref_x1 = self.build_ref_x(cfg)

    def build_ref_x(self, cfg):
        ref_fn = get_repo_path() / "data" / f"{cfg.prob.name}.pt"

        if not ref_fn.exists():
            from .dataset import get_sampler

            ref_x0 = get_sampler(cfg.prob.p0)(self.B)
            ref_x1 = get_sampler(cfg.prob.p1)(self.B)
            torch.save({"ref_x0": ref_x0, "ref_x1": ref_x1}, ref_fn)
            print(f"Saved new reference file to {ref_fn}!")
            return ref_x0, ref_x1
        else:
            ref_pt = torch.load(ref_fn, map_location="cpu")
            return ref_pt["ref_x0"], ref_pt["ref_x1"]

    def boundary_metrics(self, xs):
        B, T, D = xs.shape
        if B < self.B:
            rand_idx = torch.randint(0, B, (self.B,))
            xs = xs[rand_idx]
        elif B > self.B:
            rand_idx = torch.randperm(B)[: self.B]
            xs = xs[rand_idx]
        assert xs.shape == (self.B, T, D)

        x0, x1 = shuffle(xs[:, 0]), shuffle(xs[:, -1])
        assert x0.shape == self.ref_x0.shape == x1.shape == self.ref_x1.shape

        metrics = dict()
        from ot.sliced import sliced_wasserstein_distance
        metrics["SWD_0"] = sliced_wasserstein_distance(x0, self.ref_x0)
        metrics["SWD_1"] = sliced_wasserstein_distance(x1, self.ref_x1)

        sinkhorn = __import__('geomloss').SamplesLoss("sinkhorn", **self.sinkhorn_cfg)
        metrics["Sinkhorn_0"] = sinkhorn(x0, self.ref_x0)
        metrics["Sinkhorn_1"] = sinkhorn(x1, self.ref_x1)

        mmd = MMD_loss()
        metrics["MMD_0"] = mmd(x0, self.ref_x0)
        metrics["MMD_1"] = mmd(x1, self.ref_x1)
        return metrics

    def state_costs(self, xs):
        (B, T, D), scfg = xs.shape, self.scfg
        assert "obs" in scfg.type and scfg.obs > 0

        cost_s = scfg.obs * self.obstacle_cost(xs)
        if "ent" in scfg.type and scfg.ent > 0:
            cost_s = cost_s + scfg.ent * est_entropy_cost(xs)
        elif "cgst" in scfg.type and scfg.cgst > 0:
            cost_s = cost_s + scfg.cgst * congestion_cost(xs)

        assert cost_s.shape == (B, T)
        return cost_s

    def cost_metrics(self, xs, us):
        B, T, D = xs.shape
        assert us.shape == (B, T, D)

        scale = (0.5 / (self.sigma**2)) if self.ccfg.scale_by_sigma else 0.5
        cost_c = scale * (us**2).sum(dim=-1)
        cost_s = self.state_costs(xs)
        assert cost_c.shape == cost_s.shape == (B, T)

        metrics = dict()
        metrics["control_cost"] = cost_c.mean()
        metrics["state_cost"] = cost_s.mean()
        metrics["total_cost"] = metrics["control_cost"] + metrics["state_cost"]
        return metrics

    def __call__(self, samples):
        xs, us = cpu_everything(samples["xs"], samples["us"])
        B, T, D = xs.shape
        assert us.shape == (B, T, D)

        metrics = {}
        metrics.update(self.boundary_metrics(xs))
        metrics.update(self.cost_metrics(xs, us))
        for k, v in metrics.items():
            metrics[k] = v.item()
        return metrics
