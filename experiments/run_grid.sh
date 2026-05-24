#!/usr/bin/env bash
# Grid sweep: sync interval K x outer learning rate.
#
# The single-variable sweep (run_sweep.sh) showed that a *fixed* outer optimizer
# destabilizes short sync intervals. This grid disentangles the two: for each K
# we try several outer LRs, so we can see both
#   (a) the sensitivity (how the right outer LR depends on K), and
#   (b) the tuned communication-vs-convergence frontier (best outer LR per K).
#
# Compute budget held fixed at TOTAL_STEPS inner steps per worker; rounds = T/K.
set -euo pipefail

PY="${PY:-python3}"
WORLD="${WORLD:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-256}"
KS="${KS:-1 4 16 64}"
OUTER_LRS="${OUTER_LRS:-0.1 0.3 0.7}"
SEED="${SEED:-1337}"
OUT="${OUT:-experiments/grid}"
PORT="${PORT:-29600}"

MODEL_ARGS="--dataset tinyshakespeare --model transformer \
  --n_embd 128 --n_head 4 --n_layer 2 --ctx 128 --batch_size 32 \
  --inner_lr 1e-3 --outer_momentum 0.9 \
  --val_batches 50 --log_every 1000"

mkdir -p "$OUT"
export TORCH_DEVICE=cpu

for K in $KS; do
  ROUNDS=$(( TOTAL_STEPS / K ))
  EVAL=$(( ROUNDS / 8 )); [ "$EVAL" -lt 1 ] && EVAL=1
  for LR in $OUTER_LRS; do
    RUNDIR="$OUT/k${K}_lr${LR}"
    echo "=== K=$K outer_lr=$LR rounds=$ROUNDS -> $RUNDIR ==="
    $PY -m torch.distributed.run --nproc_per_node="$WORLD" --master_port="$PORT" \
      smol_diloco.py $MODEL_ARGS \
      --local_steps "$K" --rounds "$ROUNDS" --eval_every "$EVAL" \
      --outer_lr "$LR" --seed "$SEED" --log_dir "$RUNDIR" --ckpt "" 2>/dev/null
    PORT=$(( PORT + 1 ))
  done
done

echo "Done. Logs in $OUT/k*_lr*/metrics.jsonl"
