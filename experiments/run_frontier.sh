#!/usr/bin/env bash
# Extended frontier sweep: push the sync interval to extremes to find where the
# communication-vs-convergence frontier breaks.
#
# The grid (run_grid.sh) showed K up to 64 is fine once the outer LR is tuned.
# Here we hold the outer LR at the robust value (0.1) and a fixed compute budget,
# and extend K out to 256 -- few enough syncs that workers may drift too far for
# averaging to reconcile. Budget is large enough that even K=256 gets several
# outer rounds (budget/K), so a bad result reflects the regime, not just "1 round".
set -euo pipefail

PY="${PY:-python3}"
WORLD="${WORLD:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-768}"
KS="${KS:-1 4 16 64 128 256}"
OUTER_LR="${OUTER_LR:-0.1}"
SEED="${SEED:-1337}"
OUT="${OUT:-experiments/frontier}"
PORT="${PORT:-29700}"

MODEL_ARGS="--dataset tinyshakespeare --model transformer \
  --n_embd 128 --n_head 4 --n_layer 2 --ctx 128 --batch_size 32 \
  --inner_lr 1e-3 --outer_momentum 0.9 --outer_lr $OUTER_LR \
  --val_batches 50 --log_every 1000"

mkdir -p "$OUT"
export TORCH_DEVICE=cpu

for K in $KS; do
  ROUNDS=$(( TOTAL_STEPS / K ))
  EVAL=$(( ROUNDS / 8 )); [ "$EVAL" -lt 1 ] && EVAL=1
  RUNDIR="$OUT/k${K}"
  echo "=== K=$K outer_lr=$OUTER_LR rounds=$ROUNDS -> $RUNDIR ==="
  $PY -m torch.distributed.run --nproc_per_node="$WORLD" --master_port="$PORT" \
    smol_diloco.py $MODEL_ARGS \
    --local_steps "$K" --rounds "$ROUNDS" --eval_every "$EVAL" \
    --seed "$SEED" --log_dir "$RUNDIR" --ckpt "" 2>/dev/null
  PORT=$(( PORT + 1 ))
done

echo "Done. Logs in $OUT/k*/metrics.jsonl"
