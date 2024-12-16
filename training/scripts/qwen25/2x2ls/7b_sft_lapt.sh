#!/bin/bash
#SBATCH --job-name=lapt_qwen25_7b_sft_2x2ls_lapt
#SBATCH --mem=200G
#SBATCH --time=96:00:00


# Configs
cd /path/to/cva-merge/training/src
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
model_abbrev="Qwen2.5-7B-Instruct"
lang_code="$1"

dataset_dir="/path/to/datasets/Qwen2.5-7B-${lang_code}-lapt-madlad/"
output_dir="/path/to/models/${model_abbrev}-${lang_code}-lapt-madlad/"
model_dir="Qwen/Qwen2.5-7B-Instruct"

python main_2x2ls.py \
    --dataset_path "${dataset_dir}" \
    --output_dir "${output_dir}" \
    --logging_dir "${output_dir}" \
    --model_name_or_path "${model_dir}" \
    --tokenizer_name_or_path "${model_dir}" \
    --model_type qwen2 \
    --seed 42 \
    --evaluation_strategy no \
    --logging_steps 0.001 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_steps 30517 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns False \
    --save_strategy steps \
    --save_steps 0.25 \
    --bf16 \
    --gradient_checkpointing True \
    --is_baseline
