from functools import partial
from scipy.spatial import cKDTree
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td

from .vae import load_model, lerp, slerp
from .utils import get_repo_path


def build_state_cost(cfg):
    # ---- BBBC021 (latent vectors): FreeBridge empirical support cost ----
    if cfg.prob.name in ["bbbc021"]:
        return EmpiricalSupportStateCost(cfg.state_cost)

    if cfg.prob.name == "afhq":
        return VAEStateCost(cfg.vae)
    elif cfg.prob.name == "lidar":
        return LIDARStateCost(cfg.lidar)
    elif cfg.prob.name == "opinion":
        return OpinionStateCost(cfg.state_cost)
    else:
        # Default crowd navigation tasks (2D)
        return CrowdNavStateCost(cfg.prob.name, cfg.state_cost)


##########################################################
################### state cost functions #################
##########################################################


class ZeroStateCost(torch.nn.Module):
    """
    Returns zero cost with shape (*, T) given xt of shape (*, T, D).
    Works for any D.
    """
    def forward(self, xt, t=None, gpath=None):
        return torch.zeros(*xt.shape[:-1], device=xt.device, dtype=xt.dtype)


class EmpiricalSupportStateCost(torch.nn.Module):
    """
    Empirical support state cost for BBBC021 latent transport.

        V_t(z) = lambda * d_emp(z)^2

    where d_emp(z)^2 is approximated in each minibatch by the squared
    nearest-neighbor distance from z to the union of source/control and
    target/perturbed latent endpoints in the current batch.

    Input:  xt: (B, N, T, D), t: (T,), gpath: EndPointGaussianPath
    Output: cost: (B, N, T)
    """

    def __init__(self, scfg):
        super().__init__()
        self.weight = float(scfg.get("support_weight", 0.5))
        self.detach_bank = bool(scfg.get("detach_bank", True))
        # Avoid one huge torch.cdist allocation. If None or <=0, one shot.
        _cs = scfg.get("chunk_size", 65536)
        self.chunk_size = None if _cs is None else int(_cs)

    def forward(self, xt, t=None, gpath=None):
        if gpath is None:
            raise ValueError("EmpiricalSupportStateCost requires gpath.")
        if xt.dim() != 4:
            raise ValueError(f"Expected xt (B, N, T, D), got {tuple(xt.shape)}")

        B, N, T, D = xt.shape
        if t is not None:
            assert t.shape == (T,), f"Expected t {(T,)}, got {tuple(t.shape)}"

        # ablation shortcut: lambda == 0 -> zero cost, skip the cdist entirely
        if self.weight == 0.0:
            return torch.zeros(B, N, T, device=xt.device, dtype=xt.dtype)

        mean_xt = getattr(getattr(gpath, "mean", None), "xt", None)
        if mean_xt is not None and mean_xt.dim() == 3:
            if mean_xt.shape[0] != B or mean_xt.shape[-1] != D:
                raise ValueError(f"Shape mismatch: xt {(B,N,T,D)}, gpath.mean.xt {tuple(mean_xt.shape)}")

        # Preferred: real empirical endpoints (control + perturbed) from the
        # current minibatch, attached by pl_model. During alternating bridge-matching training
        # the gpath endpoints may be generated (not real data) after the first
        # R-step, so we use the real batch bank when available to match the
        # paper's definition (union of minibatch control + perturbed embeddings).
        if hasattr(gpath, "support_bank") and gpath.support_bank is not None:
            bank = gpath.support_bank.to(device=xt.device, dtype=xt.dtype)
            if bank.dim() != 2 or bank.shape[-1] != D:
                raise ValueError(
                    f"Expected gpath.support_bank (M, {D}), got {tuple(bank.shape)}"
                )
        else:
            # Fallback: gpath endpoints (exact only for straight-line real-endpoint paths).
            if mean_xt is None:
                raise ValueError(
                    "EmpiricalSupportStateCost needs either gpath.support_bank "
                    "or a valid gpath.mean.xt (B, T, D)."
                )
            z0 = mean_xt[:, 0, :]
            z1 = mean_xt[:, -1, :]
            bank = torch.cat([z0, z1], dim=0).to(device=xt.device, dtype=xt.dtype)

        if self.detach_bank:
            bank = bank.detach()

        query = xt.reshape(B * N * T, D)  # (BNT, D)

        if self.chunk_size is None or self.chunk_size <= 0:
            demp2 = torch.cdist(query, bank, p=2).pow(2).min(dim=1).values
        else:
            values = []
            for q in query.split(self.chunk_size, dim=0):
                values.append(torch.cdist(q, bank, p=2).pow(2).min(dim=1).values)
            demp2 = torch.cat(values, dim=0)

        cost = demp2.reshape(B, N, T)
        assert cost.shape == (B, N, T)
        return self.weight * cost


class VAEStateCost(torch.nn.Module):
    def __init__(self, vcfg):
        super().__init__()
        self.vcfg = vcfg
        self.vae = load_model(get_repo_path() / vcfg.ckpt).eval()

    @torch.no_grad()
    def recon(self, images):
        zs = self.vae.encoder(images)[0]
        return self.vae.decoder(zs)

    @torch.no_grad()
    def latent_interp(self, x0, x1, S, type="slerp"):
        """
        x0, x1: (B, D) --> xt: (B, S, D), zt: (B, S, Z)
        """
        img_size = self.vcfg.image_size
        B, D = x0.shape
        self.vae.to(x0)

        x0_img = x0.reshape(B, *img_size).to(x0)
        x1_img = x1.reshape(B, *img_size).to(x0)

        z0 = self.vae.encoder(x0_img)[0].reshape(1, B, -1)
        z1 = self.vae.encoder(x1_img)[0].reshape(1, B, -1)
        Zdim = z0.shape[-1]

        t = torch.linspace(0, 1, S).to(x0).reshape(-1, 1, 1)
        if type == "lerp":
            zt = lerp(z0, z1, t).reshape(-1, Zdim)
        elif type == "slerp":
            zt = slerp(z0, z1, t).reshape(-1, Zdim)
        else:
            raise ValueError(f"Unknown interp type: {type}")
        xt = self.vae.decoder(zt)

        recon_xt = xt.reshape(S, B, D).permute(1, 0, 2)
        recon_zt = zt.reshape(S, B, Zdim).permute(1, 0, 2)
        return recon_xt, recon_zt

    def forward(self, xt, t, recon_xt):
        """
        xt: (B, N, T, D)
        t: (T,)
        recon_xt: (B, T, D)
        ===
        (B, N, T)
        """
        B, N, T, D = xt.shape
        assert t.shape == (T,)
        assert recon_xt.shape == (B, T, D)

        recon_xt = recon_xt.reshape(B, 1, T, D).expand(-1, N, -1, -1)
        loss = ((xt - recon_xt).abs()).mean(dim=-1)  # L1
        return loss.reshape(B, N, T)


class LIDARStateCost(torch.nn.Module):
    def __init__(self, lcfg):
        super().__init__()
        import laspy

        las = laspy.read(get_repo_path() / lcfg.filename)
        self.k = lcfg.k
        self.closeness_weight = lcfg.closeness_weight
        self.height_weight = lcfg.height_weight
        self.boundary_weight = lcfg.boundary_weight
        self.lim = lcfg.lim

        mask = las.classification == 2

        x_offset, x_scale = las.header.offsets[0], las.header.scales[0]
        y_offset, y_scale = las.header.offsets[1], las.header.scales[1]
        z_offset, z_scale = las.header.offsets[2], las.header.scales[2]
        dataset = np.vstack(
            (
                las.X[mask] * x_scale + x_offset,
                las.Y[mask] * y_scale + y_offset,
                las.Z[mask] * z_scale + z_offset,
            )
        ).transpose()

        mi = dataset.min(axis=0, keepdims=True)
        ma = dataset.max(axis=0, keepdims=True)
        dataset = (dataset - mi) / (ma - mi) * [10.0, 10.0, 2.0] + [-5.0, -5.0, 0.0]

        self.dataset = dataset
        self.tree = cKDTree(dataset)

    def get_tangent_plane(self, points, temp=1e-3):
        points_np = points.detach().cpu().numpy()
        _, idx = self.tree.query(points_np, k=self.k)
        nearest_pts = self.dataset[idx]
        nearest_pts = torch.tensor(nearest_pts).to(points)

        dists = (points.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)

        w = LIDARStateCost.fit_plane(nearest_pts, weights)
        return w

    def get_tangent_proj(self, points):
        w = self.get_tangent_plane(points)
        return partial(LIDARStateCost.projection_op, w=w)

    def boundary_penalty_1d(self, x, lim=5.0):
        cost = torch.sigmoid((x - lim) / 0.1)
        cost = cost + 1 - torch.sigmoid((x + lim) / 0.1)
        return cost

    def forward(self, xt, *args, **kwargs):
        shape = xt.shape[:-1]
        assert xt.shape[-1] == 3
        N = int(np.prod(shape))
        xt = xt.reshape(N, 3)

        projx = self.get_tangent_proj(xt)
        xt_projected = projx(xt)

        closeness = (xt_projected - xt).pow(2).sum(-1).reshape(*shape)

        boundary = self.boundary_penalty_1d(xt[:, 0]) + self.boundary_penalty_1d(xt[:, 1])
        boundary = boundary.reshape(*shape)

        height = torch.exp(xt_projected[:, 2]).reshape(*shape)

        return (
            self.closeness_weight * closeness
            + self.height_weight * height
            + self.boundary_weight * boundary
        )

    @staticmethod
    def fit_plane(points, weights=None):
        D = torch.cat([points[..., :2], torch.ones_like(points[..., 2:3])], dim=-1)
        z = points[..., 2]
        if weights is not None:
            DW = D * weights
            Dtrans = DW.transpose(-1, -2)
        else:
            Dtrans = D.transpose(-1, -2)
        w = torch.linalg.solve(torch.matmul(Dtrans, D), torch.matmul(Dtrans, z.unsqueeze(-1))).squeeze(-1)
        return w

    @staticmethod
    def projection_op(x, w):
        n = torch.cat([w[..., :2], -torch.ones_like(w[..., 2:3])], dim=1)

        pn = torch.sum(x * n, dim=-1, keepdim=True)
        nn = torch.sum(n * n, dim=-1, keepdim=True)

        d = w[..., 2:3]
        projn_x = ((pn + d) / nn) * n
        return x - projn_x


class CrowdNavStateCost(torch.nn.Module):
    def __init__(self, name, scfg):
        super().__init__()
        self.name = name
        self.scfg = scfg
        self.obstacle_cost = build_obstacle_cost(name)

    def forward(self, xt, t, gpath):
        """
        xt: (*, T, D)
        t: (T,)
        ===
        cost: (*, T)
        """
        (T, D), scfg = xt.shape[-2:], self.scfg
        assert t.shape == (T,) and D == 2
        if self.obstacle_cost is None:
            raise ValueError(f"No obstacle_cost registered for prob.name='{self.name}'.")
        assert "obs" in scfg.type and scfg.obs > 0

        V = scfg.obs * self.obstacle_cost(xt)

        if "ent" in scfg.type and scfg.ent > 0:
            V = V + scfg.ent * entropy_cost(xt, t, gpath)
        elif "cgst" in scfg.type and scfg.cgst > 0:
            V = V + scfg.cgst * congestion_cost(xt)

        assert V.shape == xt.shape[:-1]
        return V


class OpinionStateCost(torch.nn.Module):
    def __init__(self, scfg):
        super().__init__()
        self.scfg = scfg

    def forward(self, xt, t, gpath):
        (T, D), scfg = xt.shape[-2:], self.scfg
        assert t.shape == (T,)

        V = zero_cost_fn(xt)
        if "ent" in scfg.type and scfg.ent > 0:
            V = V + scfg.ent * entropy_cost(xt, t, gpath)
        elif "cgst" in scfg.type and scfg.cgst > 0:
            V = V + scfg.cgst * congestion_cost(xt)

        assert V.shape == xt.shape[:-1]
        return V


def zero_cost_fn(x: torch.Tensor, *args) -> torch.Tensor:
    return torch.zeros(*x.shape[:-1], device=x.device, dtype=x.dtype)


##########################################################
################## obstacle cost functions ###############
##########################################################


def build_obstacle_cost(name):
    return {
        "gmm": obstacle_cost_gmm,
        "stunnel": obstacle_cost_stunnel,
        "vneck": obstacle_cost_vneck,
        "drunken_spider": obstacle_cost_drunken_spider,
    }.get(name, None)


def obstacle_cfg_drunken_spider():
    xys = [[-7, 0.5], [-7, -7.5]]
    widths = [14, 14]
    heights = [7, 7]
    return xys, widths, heights


def obstacle_cost_drunken_spider(xt):
    assert xt.shape[-1] == 2
    x, y = xt[..., 0], xt[..., 1]

    def cost_fn(xy, width, height):
        xbound = xy[0], xy[0] + width
        ybound = xy[1], xy[1] + height
        a = -5 * (x - xbound[0]) * (x - xbound[1])
        b = -5 * (y - ybound[0]) * (y - ybound[1])
        cost = F.softplus(a, beta=20, threshold=1) * F.softplus(b, beta=20, threshold=1)
        assert cost.shape == xt.shape[:-1]
        return cost

    return sum(cost_fn(xy, width, height) for xy, width, height in zip(*obstacle_cfg_drunken_spider()))


def obstacle_cfg_gmm():
    centers = [[6, 6], [6, -6], [-6, -6]]
    radius = 1.5
    return centers, radius


def obstacle_cfg_stunnel():
    a, b, c = 20, 1, 90
    centers = [[5, 6], [-5, -6]]
    return a, b, c, centers


def obstacle_cfg_vneck():
    c_sq = 0.36
    coef = 5
    return c_sq, coef


def obstacle_cost_gmm(xt):
    Bs, D = xt.shape[:-1], xt.shape[-1]
    assert D == 2
    xt = xt.reshape(-1, D)
    batch_xt = xt.shape[0]

    centers, radius = obstacle_cfg_gmm()

    obs1 = torch.tensor(centers[0]).repeat((batch_xt, 1)).to(xt.device)
    obs2 = torch.tensor(centers[1]).repeat((batch_xt, 1)).to(xt.device)
    obs3 = torch.tensor(centers[2]).repeat((batch_xt, 1)).to(xt.device)

    dist1 = torch.norm(xt - obs1, dim=-1)
    dist2 = torch.norm(xt - obs2, dim=-1)
    dist3 = torch.norm(xt - obs3, dim=-1)

    cost1 = F.softplus(100 * (radius - dist1), beta=1, threshold=20)
    cost2 = F.softplus(100 * (radius - dist2), beta=1, threshold=20)
    cost3 = F.softplus(100 * (radius - dist3), beta=1, threshold=20)
    return (cost1 + cost2 + cost3).reshape(*Bs)


def obstacle_cost_stunnel(xt):
    a, b, c, centers = obstacle_cfg_stunnel()
    Bs, D = xt.shape[:-1], xt.shape[-1]
    assert D == 2

    _xt = xt.reshape(-1, D)
    x, y = _xt[:, 0], _xt[:, 1]

    d = a * (x - centers[0][0]) ** 2 + b * (y - centers[0][1]) ** 2
    c1 = F.softplus(c - d, beta=1, threshold=20)

    d = a * (x - centers[1][0]) ** 2 + b * (y - centers[1][1]) ** 2
    c2 = F.softplus(c - d, beta=1, threshold=20)

    cost = (c1 + c2).reshape(*Bs)
    return cost


def obstacle_cost_vneck(xt):
    assert xt.shape[-1] == 2
    c_sq, coef = obstacle_cfg_vneck()
    xt_sq = torch.square(xt)
    d = coef * xt_sq[..., 0] - xt_sq[..., 1]
    return F.softplus(-c_sq - d, beta=1, threshold=20)


##########################################################
################ interaction cost functions ##############
##########################################################


def entropy_cost(xt, t, gpath):
    """
    xt: (B, N, T, D), t: (T,) --> (B, N, T)
    """
    B, N, T, D = xt.shape
    assert t.shape == (T,)

    mean_t = gpath.mean(t).detach()
    gamma_t = gpath.gamma(t).detach()
    assert mean_t.shape == (B, T, D)
    assert gamma_t.shape == (B, T, 1)

    normals = td.Normal(
        mean_t.reshape(B * T, D),
        gamma_t.reshape(B * T, 1) * torch.ones(B * T, D, device=gpath.device),
    )
    indep_normals = td.Independent(normals, 1)

    xxt = xt.unsqueeze(2).expand(-1, -1, B, -1, -1)
    assert xxt.shape == (B, N, B, T, D)

    log_pt_01 = indep_normals.log_prob(xxt.reshape(B * N, B * T, D)).reshape(B * N, B, T)
    pt = log_pt_01.exp().mean(dim=1)
    log_pt = pt.log().reshape(B, N, T)

    assert not torch.isnan(log_pt).any()
    assert log_pt.shape == (B, N, T)
    return log_pt


def congestion_cost(xt):
    T, D = xt.shape[-2:]
    yt = xt.reshape(-1, T, D)
    yt = yt[torch.randperm(yt.shape[0])].reshape_as(xt)
    dd = xt - yt
    dist = torch.sum(dd * dd, dim=-1)
    congestion = 2.0 / (dist + 1.0)
    assert congestion.shape == xt.shape[:-1]
    return congestion
