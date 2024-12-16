#!/bin/bash
#SBATCH --job-name=ifeval_qwen25_7b_sft_adapted
#SBATCH --output=ifeval_qwen25_7b_sft_adapted.out             
#SBATCH --time=48:00:00



# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
lang_code="$1"
model_name="your-hf-id/Qwen2.5-7B-Instruct-${lang_code}-madlad-mean-tuned"
model_abbrev="Qwen2.5-7B-Instruct"

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/cva-merge/evaluation/logs2/analysis/${model_abbrev}/" \
    --num_fewshot 0
