#!/bin/bash
#SBATCH --job-name=lighteval_gemma2_9b_adapted
#SBATCH --mem=80G
#SBATCH --time=96:00:00



# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
custom_task_script_dir="/path/to/chat-cve/evaluation/src"
log_base_dir="/path/to/chat-cve/evaluation/logs/adapted"
cache_dir="/path/to/cache"
model_abbrev="gemma-2-9b"
lang_code="$1"
declare -A lang_code_to_belebele_lang_code=(
    ["my"]="mya_Mymr"
    ["si"]="sin_Sinh"
    ["te"]="tel_Telu"
    ["am"]="amh_Ethi"
    ["gu"]="guj_Gujr"
    ["bn"]="ben_Beng"
    ["ta"]="tam_Taml"
)

# Run the script
model_name="/path/to/models/gemma-2-9b-${lang_code}-madlad-mean-tuned"
tasks=(
    "custom|mt:en2${lang_code}|0|0"
    "custom|mt:${lang_code}2en|0|0"
    "custom|sum:${lang_code}|0|1"
    "custom|sum:en|0|1"
    "lighteval|belebele_${lang_code_to_belebele_lang_code[${lang_code}]}_mcf|3|0"
    "lighteval|belebele_eng_Latn_mcf|3|0"
)

for i in $(seq 1 3); do
    for task in "${tasks[@]}"; do
        if [[ $task == "custom|"* ]]; then
            task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)
            lighteval accelerate \
                --model_args "pretrained=${model_name},dtype=bfloat16" \
                --tasks "${task}" \
                --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
                --override_batch_size 1 \
                --save_details \
                --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                --cache_dir="${cache_dir}"
        else
            task_name=$(echo $task | cut -d'|' -f2 | cut -d'_' -f1)
            lighteval accelerate \
                --model_args "pretrained=${model_name},dtype=bfloat16" \
                --tasks "${task}" \
                --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
                --override_batch_size 1 \
                --save_details \
                --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                --cache_dir="${cache_dir}"
        fi
    done
done
