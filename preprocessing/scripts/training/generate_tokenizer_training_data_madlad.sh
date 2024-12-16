#!/bin/bash
#SBATCH --job-name=generate_tokenizer_training_data
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Configs
cd /path/to/cva-merge/preprocessing/src/training/
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
lang_code="$1"

# Run the script
output_file="/path/to/datasets/madlad-${lang_code}.txt"
python generate_tokenizer_training_data_madlad.py \
    --lang_code ${lang_code} \
    --output_file ${output_file} \
    --datasets_cache_dir ${HF_DATASETS_CACHE}
