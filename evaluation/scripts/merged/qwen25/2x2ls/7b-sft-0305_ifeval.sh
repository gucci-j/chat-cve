#!/bin/bash
#SBATCH --job-name=ifeval_qwen25_7b_sft               
#SBATCH --time=48:00:00


# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
lang_code="$1"
model_name="your-hf-id/Qwen2.5-7B-Instruct-${lang_code}-madlad-mean-slerp0305-emb-special"
model_abbrev="Qwen2.5-7B-Instruct"

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/chat-cve/evaluation/logs2/merged/${model_abbrev}/" \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn


model_name="your-hf-id/Qwen2.5-7B-Instruct-${lang_code}-madlad-mean-trans0305-emb-special"
lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/chat-cve/evaluation/logs2/merged/${model_abbrev}/" \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn
