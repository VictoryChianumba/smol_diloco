#!/usr/bin/env bash
# Sync-interval sweep for smol_diloco.
#
# Holds the compute budget fixed (TOTAL_STEPS inner AdamW steps per worker) and
# varies the sync interval K (local steps between communications). Fewer syncs
# (larger K) => less communication, at some cost to convergence. The resulting
# logs feed plot_results.py to draw the communication-vs-convergence frontier.
#
# Real multi-worker training: WORLD workers, gloo backend, CPU tensors.
set -euo pipefail

PY="${PY:-python3}"
WORLD="${WORLD:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-800}"
KS="${KS:-1 4 16 64}"
SEED="${SEED:-1337}"
OUT="${OUT:-experiments/results}"
PORT="${PORT:-29555}"

# small transformer, kept identical across runs
MODEL_ARGS="--dataset tinyshakespeare --model transformer \
  --n_embd 128 --n_head 4 --n_layer 2 --ctx 128 --batch_size 32 \
  --inner_lr 1e-3 --outer_lr 0.7 --outer_momentum 0.9 \
  --val_batches 50 --log_every 1000"

mkdir -p "$OUT"
export TORCH_DEVICE=cpu

for K in $KS; do
  ROUNDS=$(( TOTAL_STEPS / K ))
  # ~10 eval points across the run, at least every round
  EVAL=$(( ROUNDS / 10 )); [ "$EVAL" -lt 1 ] && EVAL=1
  RUNDIR="$OUT/k${K}"
  echo "=== K=$K  rounds=$ROUNDS  workers=$WORLD  -> $RUNDIR ==="
  $PY -m torch.distributed.run --nproc_per_node="$WORLD" --master_port="$PORT" \
    smol_diloco.py $MODEL_ARGS \
    --local_steps "$K" --rounds "$ROUNDS" --eval_every "$EVAL" \
    --seed "$SEED" --log_dir "$RUNDIR" --ckpt "" 2>/dev/null
  PORT=$(( PORT + 1 ))
done

echo "Done. Logs in $OUT/k*/metrics.jsonl"
