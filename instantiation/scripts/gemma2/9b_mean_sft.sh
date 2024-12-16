#!/bin/bash
#SBATCH --job-name=instantiate_gemma2_sft_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00


# Configs
cd /path/to/chat-cve/instantiation/src
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
model_name_or_path="google/gemma-2-9b-it"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)

# Run the script
for lang_code in "${lang_codes[@]}"; do
    tokenizer_dir="/path/to/tokenizers/${model_abbrev}-${lang_code}-madlad/"
    output_dir="/path/to/models/${model_abbrev}-${lang_code}-madlad-mean/"
    
    python main.py \
        --source_model_name_or_path ${model_name_or_path} \
        --target_tokenizer_name_or_path ${tokenizer_dir} \
        --output_dir ${output_dir} \
        --cache_dir ${HF_HUB_CACHE} \
        --method mean

done
