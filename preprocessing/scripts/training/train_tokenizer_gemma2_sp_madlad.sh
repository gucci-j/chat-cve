#!/bin/bash
#SBATCH --job-name=train_tokenizer_sp_madlad
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Configs
cd /path/to/chat-cve/preprocessing/src/training/
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
python train_tokenizer_gemma2_sp.py \
    --corpus_path ${corpus_path} \
    --vocab_size ${vocab_size} \
    --source_tokenizer_path "/path/to/cache/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6/tokenizer.model" \
    --output_dir ${output_dir} \
    --lang_code ${lang_code} \
    --num_new_tokens ${num_new_tokens} \
    --cache_dir "${HF_DATASETS_CACHE}"
