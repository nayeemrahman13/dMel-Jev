#!/bin/bash
# Capacity sweep dispatch: 9 parallel Modal L4 runs (3 widths x 3 seeds).
# Identical protocol per run: 5 epochs, batch 32, lr 3e-4 (config.yaml defaults),
# train split only; per-run out_dir on the dmel-jev-runs volume.
set -u
cd /home/user/work/dMel-Jev
export MODAL_TOKEN_ID="${SECRET_MODAL_TOKEN_ID:?}" MODAL_TOKEN_SECRET="${SECRET_MODAL_TOKEN_SECRET:?}"
MODAL=/home/user/work/dMel-Jev/.venv/bin/modal
LOGS=/tmp/sweep_logs
mkdir -p "$LOGS"

launch() {
  local width=$1 nhead=$2 ffn=$3 seed=$4
  local name="w${width}_s${seed}"
  local start=$(date +%s)
  echo "start ${name} $(date -Is)" > "${LOGS}/${name}.log"
  "$MODAL" run dmel/training/train_modal.py \
    --arm transformer --epochs 5 --seeds 1 --seed "${seed}" \
    --d-model "${width}" --nhead "${nhead}" --dim-feedforward "${ffn}" \
    --no-generate-synthetic --data-root /data/pilot \
    --out-dir "/runs/capacity_sweep/w${width}/seed_${seed}" \
    >> "${LOGS}/${name}.log" 2>&1
  local rc=$?
  local end=$(date +%s)
  echo "RUN_EXIT=${rc} WALL_SECONDS=$((end - start)) $(date -Is)" >> "${LOGS}/${name}.log"
}

# widths: d_model nhead dim_feedforward  (head_dim preserved at 32, FFN 2.5x)
launch 256 8  640  0 & launch 256 8  640  1 & launch 256 8  640  2 &
launch 384 12 960  0 & launch 384 12 960  1 & launch 384 12 960  2 &
launch 512 16 1280 0 & launch 512 16 1280 1 & launch 512 16 1280 2 &
wait
echo ALL_RUNS_DONE
