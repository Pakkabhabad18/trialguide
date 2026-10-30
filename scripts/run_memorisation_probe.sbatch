#!/bin/bash
#SBATCH --job-name=memprobe
#SBATCH --partition=ihub
#SBATCH --account=neuro
#SBATCH --qos=neuro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Memorisation probe (T14) on the development patients: Qwen3-8B, 8-bit on one GPU, greedy.
# Submit from the repo root (the splits/ folder must hold dev.txt and test.txt):
#   mkdir -p logs
#   sbatch -w gnode097 --export=ALL,HF_HOME=/ssd_scratch/$USER/hf scripts/run_memorisation_probe.sbatch
# Smoke test on 2 patients first: add LIMIT=2 to --export.
# Writes results/memorisation_probe_dev_qwen3-8b_8bit.json and outputs/..._items.json.
# bf16 on two GPUs instead: --gres=gpu:2 and LLM_HF_8BIT=0.
MODEL=${MODEL:-Qwen/Qwen3-8B}
QUERIES=${QUERIES:-data/raw/trialgpt}
SPLIT=${SPLIT:-splits/dev.txt}
LIMIT=${LIMIT:-}
HF_ENV=${HF_ENV:-torch_env}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HF_ENV"

export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LLM_HF_8BIT=${LLM_HF_8BIT:-1}
# weight cap per GPU, used only when bf16 is split over two 12 GB GPUs
export LLM_HF_MAX_MEMORY=${LLM_HF_MAX_MEMORY:-9GiB}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv
python -m trialguide.eval.memorisation --queries "$QUERIES" --split "$SPLIT" --model "$MODEL" \
	${LIMIT:+--limit "$LIMIT"}
