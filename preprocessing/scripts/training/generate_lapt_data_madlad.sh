#!/bin/bash
#SBATCH --job-name=generate_lapt_data_madlad
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Configs
cd /path/to/cva-merge/preprocessing/src/training/
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_name_or_path="$1"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)
lang_code="$2"

# Run the script
output_dir="/path/to/datasets/${model_abbrev}-${lang_code}-madlad/"
tokenizer_name_or_path="/path/to/tokenizers/${model_abbrev}-${lang_code}-madlad/"

python generate_lapt_data_madlad.py \
    --lang_code ${lang_code} \
    --output_dir ${output_dir} \
    --datasets_cache_dir ${HF_DATASETS_CACHE} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --tokenizer_cache_dir ${HF_HUB_CACHE} \
    --num_workers 4 \
    --max_length 512
