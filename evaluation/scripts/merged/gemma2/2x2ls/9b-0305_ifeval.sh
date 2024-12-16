#!/bin/bash
#SBATCH --job-name=ifeval_gemma2_9b           
#SBATCH --time=48:00:00


# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
lang_code="$1"
model_name="your-hf-id/gemma-2-9b-${lang_code}-madlad-mean-slerp0305-emb-special"
model_abbrev="gemma-2-9b"

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/chat-cve/evaluation/logs2/merged/${model_abbrev}/" \
    --num_fewshot 0


model_name="your-hf-id/gemma-2-9b-${lang_code}-madlad-mean-trans0305-emb-special"
lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/chat-cve/evaluation/logs2/merged/${model_abbrev}/" \
    --num_fewshot 0
