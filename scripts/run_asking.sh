#!/bin/bash
# B7, A2, A3 (or other asking systems) on one GPU: rounds alternate between the checker model
# (Qwen) and the doctor model (Llama) until every conversation is finished.
# Run from the repo root inside a GPU allocation, after scripts/run_b6.sh made the manifest:
#   HF_HOME=/ssd_scratch/$USER/hf bash scripts/run_asking.sh
# Settings: SYSTEMS ("B7 A2 A3"), THRESHOLD (0.9, stand-in for lambda), OUT (outputs/asking),
# LLM_CACHE (shared with other runs on the same patients, so their checks are reused).
SYSTEMS=${SYSTEMS:-"B7 A2 A3"}
THRESHOLD=${THRESHOLD:-0.9}
OUT=${OUT:-outputs/asking}
MANIFEST=${MANIFEST:-outputs/b6/data/manifest.json}
MAX_ROUNDS=${MAX_ROUNDS:-12}
HF_ENV=${HF_ENV:-torch_env}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HF_ENV"
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LLM_BACKEND=hf LLM_HF_8BIT=1 LLM_HF_BATCH=${LLM_HF_BATCH:-8} LLM_MAX_TOKENS=${LLM_MAX_TOKENS:-1024}
export LLM_CACHE=${LLM_CACHE:-$OUT/llm_cache.jsonl} LLM_LOG=${LLM_LOG:-$OUT/llm_calls.jsonl}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
echo "systems=$SYSTEMS threshold=$THRESHOLD commit=$(git rev-parse --short HEAD) start=$(date -Is)"

for ((round = 1; round <= MAX_ROUNDS; round++)); do
	if ((round % 2)); then MODEL=Qwen/Qwen3-8B; else MODEL=meta-llama/Llama-3.1-8B-Instruct; fi
	echo "== round $round ($MODEL) $(date -Is)"
	LLM_ONLY_MODEL=$MODEL python -m trialguide.baselines.run_asking --manifest "$MANIFEST" \
		--systems $SYSTEMS --threshold "$THRESHOLD" --out "$OUT" ${ALLOW_TEST:+--allow-test}
	if [ -f "$OUT/summary.json" ] && python -c "import json,sys; sys.exit(bool(json.load(open('$OUT/status.json'))['waiting']))"; then
		echo "all conversations finished end=$(date -Is)"
		exit 0
	fi
done
echo "stopped after $MAX_ROUNDS rounds with conversations still waiting"
exit 1
