#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run random vs sequential block layout benchmark N times each,
# dropping page cache and sleeping between runs for stable measurements.
#
# Usage: bash run_block_layout_bench.sh [RUNS] [ITERS]
#   RUNS  : number of repetitions per layout (default: 5)
#   ITERS : benchmark iterations per run     (default: 20)

RUNS=${1:-5}
ITERS=${2:-20}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/../../../.venv/bin/activate"
DROP_SH="/home/beom115/quicksilver/System-configs/linux/drop.sh"

BENCH_ARGS=(
  --batch-size 32
  --q-len-min 1 --q-len-max 1
  --kv-len-min 512 --kv-len-max 512
  --num-query-heads 32 --num-kv-heads 8
  --head-size 128 --block-size 128
  --iters "$ITERS"
)

# shellcheck source=/dev/null
source "$VENV"

drop_and_sleep() {
  sudo bash "$DROP_SH" && sleep 5
}

echo "========================================"
echo " Block layout benchmark  (runs=$RUNS, iters=$ITERS)"
echo "========================================"

echo ""
echo "--- random ---"
for i in $(seq 1 "$RUNS"); do
  drop_and_sleep
  result=$(python "$SCRIPT_DIR/benchmark_cpu_attn.py" \
    "${BENCH_ARGS[@]}" --block-layout random 2>&1 | grep "median")
  echo "  Run $i: $result"
done

echo ""
echo "--- sequential ---"
for i in $(seq 1 "$RUNS"); do
  drop_and_sleep
  result=$(python "$SCRIPT_DIR/benchmark_cpu_attn.py" \
    "${BENCH_ARGS[@]}" --block-layout sequential 2>&1 | grep "median")
  echo "  Run $i: $result"
done

echo ""
echo "done."
