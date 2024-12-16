#!/bin/bash
#SBATCH --job-name=train_tokenizer_madlad
#SBATCH --mem=64G
#SBATCH --time=96:00:00

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
vocab_size=50000
num_new_tokens=10000

# Run the script
corpus_path="/path/to/datasets/madlad-${lang_code}.txt"
output_dir="/path/to/tokenizers/${model_abbrev}-${lang_code}-madlad/"
mkdir $output_dir
python train_tokenizer.py \
    --corpus_path ${corpus_path} \
    --vocab_size ${vocab_size} \
    --output_dir ${output_dir} \
    --lang_code ${lang_code} \
    --num_new_tokens ${num_new_tokens} \
    --datasets_cache_dir "${HF_DATASETS_CACHE}" \
    --hub_cache_dir "${HF_HUB_CACHE}" \
    --model_name_or_path ${model_name_or_path}
