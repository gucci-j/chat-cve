Evaluation
===

Here are the examples of how to evaluate a model on LightEval and LM-Eval-Harness.

## Evaluation on LightEval
The following is an example of a bash script that can be used to evaluate a model on LightEval.
The example uses "Qwen/Qwen2.5-7B-Instruct" model to evaluate on the all the tasks used in the paper except for IFEval.
You can find scripts for other models in the [`scripts`](./scripts/) directory.

```bash
#!/bin/bash
#SBATCH --job-name=lighteval_qwen25_7b_sft
#SBATCH --mem=80G
#SBATCH --time=96:00:00

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
custom_task_script_dir="/path/to/chat-cve/evaluation/src"
log_base_dir="/path/to/chat-cve/evaluation/logs/source/sft"
cache_dir="/path/to/cache"
model_name="Qwen/Qwen2.5-7B-Instruct"
model_abbrev="Qwen2.5-7B-Instruct"
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
tasks=(
    "custom|mt:en2${lang_code}|0|0"
    "custom|mt:${lang_code}2en|0|0"
    "custom|sum:${lang_code}|0|1"
    "lighteval|belebele_${lang_code_to_belebele_lang_code[${lang_code}]}_mcf|3|0"
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
                --cache_dir="${cache_dir}" \
                --use_chat_template 
        else
            task_name=$(echo $task | cut -d'|' -f2 | cut -d'_' -f1)
            lighteval accelerate \
                --model_args "pretrained=${model_name},dtype=bfloat16" \
                --tasks "${task}" \
                --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
                --override_batch_size 1 \
                --save_details \
                --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                --cache_dir="${cache_dir}" \
                --use_chat_template 
        fi
    done
done


# Run the script
tasks=(
    "custom|sum:en|0|1"
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
                --max_samples 500 \
                --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                --cache_dir="${cache_dir}" \
                --use_chat_template 
        else
            if [[ $task == "lighteval|belebele_eng_Latn_mcf|3|0" ]]; then
                task_name=$(echo $task | cut -d'|' -f2 | cut -d'_' -f1)
                lighteval accelerate \
                    --model_args "pretrained=${model_name},dtype=bfloat16" \
                    --tasks "${task}" \
                    --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
                    --override_batch_size 1 \
                    --save_details \
                    --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                    --cache_dir="${cache_dir}" \
                    --use_chat_template 
            else
                task_name=$(echo $task | cut -d'|' -f2 | tr ':' '_')
                lighteval accelerate \
                    --model_args "pretrained=${model_name},dtype=bfloat16" \
                    --tasks "${task}" \
                    --override_batch_size 1 \
                    --save_details \
                    --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                    --cache_dir="${cache_dir}" \
                    --use_chat_template 
            fi
        fi
    done
done

tasks=(
    "harness|bbh:boolean_expressions|3|0"
    "harness|bbh:causal_judgment|3|0"
    "harness|bbh:date_understanding|3|0"
    "harness|bbh:disambiguation_qa|3|0"
    #harness|bbh:dyck_languages|3|0
    "harness|bbh:formal_fallacies|3|0"
    "harness|bbh:geometric_shapes|3|0"
    "harness|bbh:hyperbaton|3|0"
    "harness|bbh:logical_deduction_five_objects|3|0"
    "harness|bbh:logical_deduction_seven_objects|3|0"
    "harness|bbh:logical_deduction_three_objects|3|0"
    "harness|bbh:movie_recommendation|3|0"
    #harness|bbh:multistep_arithmetic_two|3|0
    "harness|bbh:navigate|3|0"
    "harness|bbh:object_counting|3|0"
    "harness|bbh:penguins_in_a_table|3|0"
    "harness|bbh:reasoning_about_colored_objects|3|0"
    "harness|bbh:ruin_names|3|0"
    "harness|bbh:salient_translation_error_detection|3|0"
    "harness|bbh:snarks|3|0"
    "harness|bbh:sports_understanding|3|0"
    "harness|bbh:temporal_sequences|3|0"
    "harness|bbh:tracking_shuffled_objects_five_objects|3|0"
    "harness|bbh:tracking_shuffled_objects_seven_objects|3|0"
    "harness|bbh:tracking_shuffled_objects_three_objects|3|0"
    "harness|bbh:web_of_lies|3|0"
    #harness|bbh:word_sorting|3|0
    "leaderboard|mmlu:abstract_algebra|5|0"
    "leaderboard|mmlu:anatomy|5|0"
    "leaderboard|mmlu:astronomy|5|0"
    "leaderboard|mmlu:business_ethics|5|0"
    "leaderboard|mmlu:clinical_knowledge|5|0"
    "leaderboard|mmlu:college_biology|5|0"
    "leaderboard|mmlu:college_chemistry|5|0"
    "leaderboard|mmlu:college_computer_science|5|0"
    "leaderboard|mmlu:college_mathematics|5|0"
    "leaderboard|mmlu:college_medicine|5|0"
    "leaderboard|mmlu:college_physics|5|0"
    "leaderboard|mmlu:computer_security|5|0"
    "leaderboard|mmlu:conceptual_physics|5|0"
    "leaderboard|mmlu:econometrics|5|0"
    "leaderboard|mmlu:electrical_engineering|5|0"
    "leaderboard|mmlu:elementary_mathematics|5|0"
    "leaderboard|mmlu:formal_logic|5|0"
    "leaderboard|mmlu:global_facts|5|0"
    "leaderboard|mmlu:high_school_biology|5|0"
    "leaderboard|mmlu:high_school_chemistry|5|0"
    "leaderboard|mmlu:high_school_computer_science|5|0"
    "leaderboard|mmlu:high_school_european_history|5|0"
    "leaderboard|mmlu:high_school_geography|5|0"
    "leaderboard|mmlu:high_school_government_and_politics|5|0"
    "leaderboard|mmlu:high_school_macroeconomics|5|0"
    "leaderboard|mmlu:high_school_mathematics|5|0"
    "leaderboard|mmlu:high_school_microeconomics|5|0"
    "leaderboard|mmlu:high_school_physics|5|0"
    "leaderboard|mmlu:high_school_psychology|5|0"
    "leaderboard|mmlu:high_school_statistics|5|0"
    "leaderboard|mmlu:high_school_us_history|5|0"
    "leaderboard|mmlu:high_school_world_history|5|0"
    "leaderboard|mmlu:human_aging|5|0"
    "leaderboard|mmlu:human_sexuality|5|0"
    "leaderboard|mmlu:international_law|5|0"
    "leaderboard|mmlu:jurisprudence|5|0"
    "leaderboard|mmlu:logical_fallacies|5|0"
    "leaderboard|mmlu:machine_learning|5|0"
    "leaderboard|mmlu:management|5|0"
    "leaderboard|mmlu:marketing|5|0"
    "leaderboard|mmlu:medical_genetics|5|0"
    "leaderboard|mmlu:miscellaneous|5|0"
    "leaderboard|mmlu:moral_disputes|5|0"
    "leaderboard|mmlu:moral_scenarios|5|0"
    "leaderboard|mmlu:nutrition|5|0"
    "leaderboard|mmlu:philosophy|5|0"
    "leaderboard|mmlu:prehistory|5|0"
    "leaderboard|mmlu:professional_accounting|5|0"
    "leaderboard|mmlu:professional_law|5|0"
    "leaderboard|mmlu:professional_medicine|5|0"
    "leaderboard|mmlu:professional_psychology|5|0"
    "leaderboard|mmlu:public_relations|5|0"
    "leaderboard|mmlu:security_studies|5|0"
    "leaderboard|mmlu:sociology|5|0"
    "leaderboard|mmlu:us_foreign_policy|5|0"
    "leaderboard|mmlu:virology|5|0"
    "leaderboard|mmlu:world_religions|5|0"
)

    for task in "${tasks[@]}"; do
        task_name=$(echo $task | cut -d'|' -f2)
        lighteval accelerate \
            --model_args "pretrained=${model_name},dtype=bfloat16" \
            --tasks "${task}" \
            --override_batch_size 1 \
            --save_details \
            --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
            --cache_dir="${cache_dir}" \
            --use_chat_template
    done

```

## Evaluation on LM-Eval-Harness

The following is an example of a bash script that can be used to evaluate a model on LM-Eval-Harness. 
The example uses "Qwen/Qwen2.5-7B-Instruct" model to evaluate on the IFEval dataset.

```bash
#!/bin/bash
#SBATCH --job-name=ifeval_qwen25_7b_sft
#SBATCH --output=ifeval_qwen25_7b_sft.out                      
#SBATCH --time=48:00:00

# Configs
export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"

model_name="Qwen/Qwen2.5-7B-Instruct"
model_abbrev="Qwen2.5-7B-Instruct"

lm-eval --model hf \
    --model_args=pretrained=${model_name},dtype=bfloat16 \
    --tasks=leaderboard_ifeval \
    --batch_size=1 \
    --output_path="/path/to/chat-cve/evaluation/logs2/source/${model_abbrev}/" \
    --num_fewshot 0 \
    --apply_chat_template \
    --fewshot_as_multiturn
```
