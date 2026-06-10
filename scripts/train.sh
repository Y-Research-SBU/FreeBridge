#!/usr/bin/env bash
# FreeBridge (BBBC021) support-weight ablation.
set -e

python train.py experiment=bbbc021 state_cost.support_weight=0.0
python train.py experiment=bbbc021 state_cost.support_weight=0.3
python train.py experiment=bbbc021 state_cost.support_weight=0.5   # default
