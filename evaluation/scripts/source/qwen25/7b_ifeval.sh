#!/bin/bash
#SBATCH --job-name=ifeval_qwen25_7b
#SBATCH --output=ifeval_qwen25_7b.out  
#SBATCH --time=48:00:00

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"

model_name="Qwen/Qwen2.5-7B"
model_abbrev="Qwen2.5-7B"

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/cva-merge/evaluation/logs2/source/${model_abbrev}/" \
    --num_fewshot 0
