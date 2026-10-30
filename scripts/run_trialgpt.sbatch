#!/bin/bash
#SBATCH --job-name=trialgpt
#SBATCH --partition=ihub
#SBATCH --account=neuro
#SBATCH --qos=neuro
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

# TrialGPT end to end (matching -> aggregation -> ranking -> evaluation) on an
# open model, for the IIIT ada cluster. Submit from the repo root after
# scripts/download_data.sh; every setting below can be overridden with
# sbatch --export=ALL,VAR=value. Example (smoke test, model cache on gnode097's
# local scratch):
#   mkdir -p logs
#   sbatch -w gnode097 --export=ALL,HF_HOME=/ssd_scratch/$USER/hf scripts/run_trialgpt.sbatch
# Full cohort instead of the smoke subset: add N_PATIENTS=,MAX_TRIALS=,TAG= to --export.
# It can also run inside an interactive GPU session: bash scripts/run_trialgpt.sbatch
#
# BACKEND=hf runs transformers in-process; BACKEND=openai serves the model with
# vLLM, which is much faster once vLLM is installed.
BACKEND=${BACKEND:-hf}
MODEL=${MODEL:-Qwen/Qwen3-8B}
TP=${TP:-2}
MAX_LEN=${MAX_LEN:-16384}
CORPUS=${CORPUS:-sigir}
N_PATIENTS=${N_PATIENTS-3}
MAX_TRIALS=${MAX_TRIALS-5}
TAG=${TAG-_smoke}
HF_ENV=${HF_ENV:-torch_env}
VLLM_ENV=${VLLM_ENV:-trialgpt}
OUT_DIR=${OUT_DIR:-outputs/trialgpt}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# TrialGPT data as put there by scripts/download_data.sh
export TRIALGPT_DATA=${TRIALGPT_DATA:-data/raw/trialgpt}
source "$(conda info --base)/etc/profile.d/conda.sh"

export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LLM_BACKEND=$BACKEND
export LLM_CACHE=${LLM_CACHE:-outputs/llm_cache.jsonl}
export LLM_LOG=${LLM_LOG:-outputs/llm_calls.jsonl}

if [ "$BACKEND" = hf ]; then
	conda activate "$HF_ENV"
	export LLM_HF_BATCH=${LLM_HF_BATCH:-2}
	# weight cap per GPU: Qwen3-8B (16.4 GB) splits ~evenly over two 12 GB GPUs,
	# leaving ~3 GB each for the KV cache
	export LLM_HF_MAX_MEMORY=${LLM_HF_MAX_MEMORY:-9GiB}
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	WORKERS=${WORKERS:-8}
else
	conda activate "$VLLM_ENV"
	PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))
	export OPENAI_BASE_URL=http://localhost:$PORT/v1
	WORKERS=${WORKERS:-32}
	mkdir -p logs

	vllm serve "$MODEL" --port $PORT --tensor-parallel-size $TP \
		--max-model-len $MAX_LEN --gpu-memory-utilization 0.92 --seed 0 \
		> logs/vllm_${SLURM_JOB_ID:-local}.log 2>&1 &
	VLLM_PID=$!
	trap "kill $VLLM_PID" EXIT

	until curl -sf http://localhost:$PORT/health > /dev/null; do
		kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died; see logs/vllm_${SLURM_JOB_ID:-local}.log"; exit 1; }
		sleep 10
	done
fi
echo "backend=$BACKEND model=$MODEL host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

SUBSET=()
[ -n "$N_PATIENTS" ] && SUBSET+=(--n_patients "$N_PATIENTS")
[ -n "$MAX_TRIALS" ] && SUBSET+=(--max_trials_per_label "$MAX_TRIALS")
NAME=${CORPUS}_${MODEL//\//_}${TAG}
M=trialguide.baselines.trialgpt

python -m $M.run_matching "$CORPUS" "$MODEL" "${SUBSET[@]}" --workers $WORKERS --tag "$TAG" --out_dir "$OUT_DIR"
python -m $M.run_aggregation "$CORPUS" "$MODEL" "$OUT_DIR/matching_results_$NAME.json" --workers $WORKERS --tag "$TAG" --out_dir "$OUT_DIR"
python -m $M.rank_results "$OUT_DIR/matching_results_$NAME.json" "$OUT_DIR/aggregation_results_$NAME.json" "$OUT_DIR/ranking_$NAME.json" > /dev/null
python -m $M.evaluate "$CORPUS" "$OUT_DIR/ranking_$NAME.json" "$OUT_DIR/matching_results_$NAME.json" | tee "$OUT_DIR/eval_$NAME.json"
echo "end=$(date -Is)"
