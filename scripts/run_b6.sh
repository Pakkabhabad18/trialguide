#!/bin/bash
# B6 (T16): TrialGPT on the full and on the masked notes of a few development patients.
# Run from the repo root inside a GPU allocation (8-bit Qwen3-8B fits one 12 GB GPU):
#   HF_HOME=/ssd_scratch/$USER/hf bash scripts/run_b6.sh
# or as a job: sbatch --gres=gpu:1 -p ihub -A neuro --qos=neuro -t 12:00:00 scripts/run_b6.sh
# Settings: N (patients, 5), PER_LABEL (trials per expert label, 2), FRACTION (rule mask, 0.4).
MODEL=${MODEL:-Qwen/Qwen3-8B}
N=${N:-5}
PER_LABEL=${PER_LABEL:-2}
FRACTION=${FRACTION:-0.4}
OUT=${OUT:-outputs/b6}
HF_ENV=${HF_ENV:-torch_env}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HF_ENV"

export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LLM_BACKEND=hf LLM_HF_8BIT=${LLM_HF_8BIT:-1} LLM_HF_BATCH=${LLM_HF_BATCH:-4}
export LLM_CACHE=${LLM_CACHE:-$OUT/llm_cache.jsonl} LLM_LOG=${LLM_LOG:-$OUT/llm_calls.jsonl}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=trialguide.baselines.trialgpt
echo "model=$MODEL 8bit=$LLM_HF_8BIT host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"

python -m trialguide.baselines.b6 --raw data/raw --out "$OUT/data" --n "$N" --per-label "$PER_LABEL" --fraction "$FRACTION"

# One Python process per step, so the model loads once per step; full note first (the reference)
for COND in full "rule$(python -c "print(round($FRACTION * 100))")"; do
	for CORPUS in $(ls "$OUT/data/$COND"); do
		export TRIALGPT_DATA=$OUT/data/$COND
		R=$OUT/results/$COND
		NAME=${CORPUS}_${MODEL//\//_}
		echo "== $COND $CORPUS $(date -Is)"
		python -m $M.run_matching "$CORPUS" "$MODEL" --workers 8 --out_dir "$R"
		python -m $M.run_aggregation "$CORPUS" "$MODEL" "$R/matching_results_$NAME.json" --workers 8 --out_dir "$R"
		python -m $M.rank_results "$R/matching_results_$NAME.json" "$R/aggregation_results_$NAME.json" "$R/ranking_$NAME.json" > /dev/null
		python -m $M.evaluate "$CORPUS" "$R/ranking_$NAME.json" "$R/matching_results_$NAME.json" > "$R/eval_$NAME.json"
		cat "$R/eval_$NAME.json"
	done
done
echo "end=$(date -Is)"
