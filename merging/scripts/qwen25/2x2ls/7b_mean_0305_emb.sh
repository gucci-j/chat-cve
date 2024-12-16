#!/bin/bash
#SBATCH --job-name=merge_adapted_2x2ls_qwen25_7b
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Run the script
cd /path/to/chat-cve/merging/src/

model_abbrev="Qwen2.5-7B"
lang_codes=(
    "am"
    "bn"
    "gu"
    "my"
    "si"
    "ta"
    "te"
)

for lang_code in "${lang_codes[@]}"
do
    #####
    # 2x2LS
    #####
    python main.py \
        --model_src_name_or_path "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-tuned/" \
        --model_tgt_name_or_path "Qwen/Qwen2.5-7B" \
        --tokenizer_src_name_or_path "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-tuned/" \
        --pipeline add_transition copy_emb \
        --transition_indices 0 1 -2 -1 \
        --transition_rates 0.3 0.5 0.5 0.3 \
        --transition_method linear \
        --cache_dir "/path/to/cache" \
        --output_dir "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-trans0305-emb"
        
done
