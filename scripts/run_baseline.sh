#!/usr/bin/env bash
# Collect runs with the official harness, then ingest + label them.
#
# Usage: scripts/run_baseline.sh <submission_dir> <model_name> <tasks.jsonl> <snapshots_dir> [out_dir]
#
# The swegemma flags below come from the third-party copy of the harness spec.
# Check `swegemma eval --help` in week 1 and fix them if they differ.
set -euo pipefail

SUB=${1:?submission dir (agent.yaml etc.)}
MODEL=${2:?model name for the records, e.g. gemma-4-31b-w4a16}
TASKS=${3:?tasks.jsonl}
SNAPS=${4:?snapshots dir}
OUT=${5:-runs/$(basename "$SUB")-$(date +%Y%m%d-%H%M)}

mkdir -p "$OUT"
CONFIG_HASH=$(find "$SUB" -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.md' \) -print0 \
  | sort -z | xargs -0 cat | sha256sum | cut -c1-12)
echo "config_hash=$CONFIG_HASH" | tee "$OUT/meta.txt"
cp -r "$SUB" "$OUT/submission_snapshot"

time swegemma eval \
  --tasks "$TASKS" \
  --snapshots-dir "$SNAPS" \
  --submission-dir "$SUB" \
  --results-dir "$OUT/results" \
  --sandbox subprocess \
  --max-tool-calls 50 \
  --concurrency 2

faultlines ingest --results "$OUT/results" --tasks "$TASKS" --model "$MODEL" \
  --config-hash "$CONFIG_HASH" -o "$OUT/runs.jsonl"
faultlines label "$OUT/runs.jsonl" -o "$OUT/labeled.jsonl"
faultlines summarize "$OUT/labeled.jsonl" | tee "$OUT/summary.md"
