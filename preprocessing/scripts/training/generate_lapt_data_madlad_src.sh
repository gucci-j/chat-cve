#!/bin/bash
#SBATCH --job-name=generate_lapt_data_madlad_src
#SBATCH --mem=64G
#SBATCH --time=96:00:00

# Configs
cd /path/to/chat-cve/preprocessing/src/training/
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
lang_codes=(
    "am"
    "bn"
    "gu"
    "my"
    "si"
    "ta"
    "te"
)
model_name_or_path="$1"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)

# Run the script
for lang_code in "${lang_codes[@]}"; do
    output_dir="/path/to/datasets/${model_abbrev}-${lang_code}-lapt-madlad/"

    python generate_lapt_data_madlad.py \
        --lang_code ${lang_code} \
        --output_dir ${output_dir} \
        --datasets_cache_dir ${HF_DATASETS_CACHE} \
        --tokenizer_name_or_path ${model_name_or_path} \
        --tokenizer_cache_dir ${HF_HUB_CACHE} \
        --num_workers 4 \
        --max_length 512

done
