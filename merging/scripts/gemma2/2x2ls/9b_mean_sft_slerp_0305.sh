#!/bin/bash
#SBATCH --job-name=merge_adapted_2x2ls_gemma2_9b_sft
#SBATCH --mem=64G
#SBATCH --time=24:00:00


# Run the script
cd /path/to/chat-cve/merging/src/

model_abbrev="gemma-2-9b-it"
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
        --model_tgt_name_or_path "google/gemma-2-9b-it" \
        --tokenizer_src_name_or_path "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-tuned/" \
        --pipeline add_transition copy_emb \
        --consider_special_tokens \
        --transition_indices 0 1 -2 -1 \
        --transition_rates 0.3 0.5 0.5 0.3 \
        --transition_method slerp \
        --cache_dir "/path/to/cache" \
        --output_dir "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-slerp0305-emb-special"
        
done
