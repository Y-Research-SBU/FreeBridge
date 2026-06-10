import os
import sys
import json
import logging
import inspect
from glob import glob
from datetime import datetime
from typing import Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from freebridge.dataset import get_dist_boundary
from freebridge.pl_model import FreeBridgeLitModule

try:
    import colored_traceback.always  # noqa: F401
except Exception:
    pass

torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


def seed_all(seed: int):
    """Robust seeding across Lightning versions."""
    try:
        pl.seed_everything(seed, workers=True)
        return
    except Exception:
        pass

    try:
        from lightning_fabric.utilities.seed import seed_everything
        seed_everything(seed, workers=True)
        return
    except Exception:
        pass

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_slurm_plugin():
    """
    Lightning compatibility: SLURMEnvironment location differs across versions.
    Return a plugin instance if available + detected, else None.
    """
    try:
        env_cls = pl.plugins.environments.SLURMEnvironment  # type: ignore[attr-defined]
        plugin = env_cls(auto_requeue=False)
        if plugin.detect():
            return plugin
        return None
    except Exception:
        return None


def should_log_boundary(cfg: DictConfig) -> bool:
    """
    Avoid plotting for custom problems (e.g., bbbc021 latents) since it often
    requires cfg.plot and toy/image semantics.
    Only enable for known problems or if cfg.plot exists.
    """
    prob_name = str(cfg.prob.name)
    if prob_name in {"toy", "opinion", "lidar", "afhq"}:
        return True
    try:
        _ = cfg.plot  # noqa: F841
        return True
    except Exception:
        return False


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    hydra_config = HydraConfig.get()
    nnodes = hydra_config.launcher.get("nodes", 1)
    print("nnodes", nnodes)

    if cfg.get("seed", None) is not None:
        seed_all(int(cfg.seed))

    # Print cfg (helps debug Hydra overrides)
    print(cfg)

    # Device info
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} CUDA devices.")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"{props.name}\t Memory: {props.total_memory / (1024**3):.2f}GB")

    # SLURM env dump
    keys = [
        "SLURM_NODELIST",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))

    # Save invoked command for reproducibility
    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)

    log.info(f"CWD: {os.getcwd()}")

    # Construct model
    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    # config sanity checks
    if int(cfg.nfe) + 1 < int(cfg.csoc.T_mean):
        raise ValueError(f"nfe+1 ({cfg.nfe}+1) must be >= csoc.T_mean ({cfg.csoc.T_mean})")

    model = FreeBridgeLitModule(cfg, p0, p1, p0_val, p1_val)

    # Optional plotting/logging (guarded)
    if should_log_boundary(cfg):
        try:
            model.log_boundary(p0, p1, p0_val, p1_val)
        except Exception as e:
            log.warning(f"Skip log_boundary due to error: {repr(e)}")

    if str(cfg.prob.name) == "opinion":
        try:
            model.log_basedrift(p0)
        except Exception as e:
            log.warning(f"Skip log_basedrift due to error: {repr(e)}")

    # ---- Callbacks ----
    # 1) Always save per-epoch + last
    ckpt_all = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch-{epoch:03d}_step-{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,  # save all
        save_last=True,
        every_n_epochs=1,
        verbose=True,
    )

    # 2) "Best" checkpoint: DO NOT monitor val_loss unless you actually log it.
    # Your current run only has train/* keys, so monitor train/loss_epoch.
    ckpt_best = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best",
        auto_insert_metric_name=False,
        monitor="train/loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )

    callbacks = [
        ckpt_all,
        ckpt_best,
        LearningRateMonitor(),
    ]

    # ---- Loggers ----
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()

    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.get("use_wandb", False):
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.prob.name}_{now}",
                project="FreeBridge",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )

    # Optional SLURM plugin
    slurm_plugin = maybe_slurm_plugin()

    # ---- Build Trainer kwargs with strict compatibility guards ----
    sig = inspect.signature(pl.Trainer.__init__)

    trainer_kwargs = dict(
        max_epochs=cfg.optim.max_epochs,
        accelerator=("gpu" if n_gpus > 0 else "cpu"),
        logger=loggers,
        num_nodes=nnodes,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,  # safer across versions
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
    )

    if "devices" in sig.parameters:
        trainer_kwargs["devices"] = n_gpus if n_gpus > 0 else 1

    if n_gpus > 1 and "strategy" in sig.parameters:
        trainer_kwargs["strategy"] = "ddp"

    if slurm_plugin is not None and "plugins" in sig.parameters:
        trainer_kwargs["plugins"] = slurm_plugin

    if "replace_sampler_ddp" in sig.parameters:
        trainer_kwargs["replace_sampler_ddp"] = False

    # The bridge optimizer runs the R-step inside validation_step via backward();
    # disable Lightning's inference_mode so gradients are allowed there.
    if "inference_mode" in sig.parameters:
        trainer_kwargs["inference_mode"] = False

    trainer = pl.Trainer(**trainer_kwargs)

    # ---- Resume logic ----
    # Do NOT override an explicit cfg.resume.
    checkpoint: Optional[str] = cfg.get("resume", None)

    # If user did NOT pass resume, allow resuming latest in *this* working dir.
    if checkpoint is None:
        last_ckpt = "checkpoints/last.ckpt"
        if os.path.exists(last_ckpt):
            checkpoint = last_ckpt
        else:
            checkpoints = glob("checkpoints/**/*.ckpt", recursive=True)
            if len(checkpoints) > 0:
                checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    # ---- Fit ----
    trainer.fit(model, ckpt_path=checkpoint)

    # Print available metric keys (helps future monitor decisions)
    try:
        keys = list(trainer.callback_metrics.keys())
        log.info(f"callback_metrics keys: {keys}")
    except Exception:
        pass

    # Save metrics.json
    metric_dict = trainer.callback_metrics
    out = {}
    for k, v in metric_dict.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = str(v)

    with open("metrics.json", "w") as fout:
        print(json.dumps(out), file=fout)

    return out


if __name__ == "__main__":
    main()