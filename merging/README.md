Model Merging
===

Here are the examples to merge models with linear or SLERP methods.

The scripts for Llama 3.1 and Gemma 2 are available in the [`scripts`](./scripts/) directory.

The naming convention for the scripts for Llama 3.1 and Gemma 2 is the same as the scripts for Qwen2.5.

NOTE: The scripts are provided as examples. You may need to modify the scripts to fit your environment. Also, we assume the use of the Slurm workload manager.

## Qwen2.5 (Chat)

### SLERP
#### Merge
[`7b_mean_sft_slerp_0305_trans_only.sh`](./scripts/qwen25/2x2ls/7b_mean_sft_slerp_0305_trans_only.sh):  
```bash
#!/bin/bash
#SBATCH --job-name=merge_adapted_2x2ls_qwen25_7b_sft
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Run the script
cd /path/to/chat-cve/merging/src/

model_abbrev="Qwen2.5-7B-Instruct"
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
        --model_tgt_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
        --tokenizer_src_name_or_path "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-tuned/" \
        --pipeline add_transition copy_emb \
        --transition_indices 0 1 -2 -1 \
        --transition_rates 0.3 0.5 0.5 0.3 \
        --transition_method slerp \
        --cache_dir "/path/to/cache" \
        --output_dir "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-slerp0305-emb"
        
done

```

#### Copy+Merge
[`7b_mean_sft_slerp_0305.sh`](./scripts/qwen25/2x2ls/7b_mean_sft_slerp_0305.sh):  
```bash
#!/bin/bash
#SBATCH --job-name=merge_adapted_2x2ls_qwen25_7b_sft
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Run the script
cd /path/to/chat-cve/merging/src/

model_abbrev="Qwen2.5-7B-Instruct"
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
        --model_tgt_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
        --tokenizer_src_name_or_path "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-tuned/" \
        --pipeline add_transition copy_emb \
        --consider_special_tokens \
        --transition_indices 0 1 -2 -1 \
        --transition_rates 0.3 0.5 0.5 0.3 \
        --transition_method slerp \
        --cache_dir "/path/to/cache" \
        --output_dir "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-slerp0305-emb-special"
        
done
```

### Linear
#### Copy+Merge
[`7b_mean_sft_0305.sh`](./scripts/qwen25/2x2ls/7b_mean_sft_0305.sh):  
```bash
#!/bin/bash
#SBATCH --job-name=merge_adapted_2x2ls_qwen25_7b_sft
#SBATCH --mem=64G
#SBATCH --time=24:00:00


# Run the script
cd /path/to/chat-cve/merging/src/

model_abbrev="Qwen2.5-7B-Instruct"
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
        --model_tgt_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
        --tokenizer_src_name_or_path "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-tuned/" \
        --pipeline add_transition copy_emb \
        --consider_special_tokens \
        --transition_indices 0 1 -2 -1 \
        --transition_rates 0.3 0.5 0.5 0.3 \
        --transition_method linear \
        --cache_dir "/path/to/cache" \
        --output_dir "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-trans0305-emb-special"
        
done
```


## Qwen2.5 (Base)
### SLERP
#### Merge
[`7b_mean_slerp_0305_emb.sh`](./scripts/qwen25/2x2ls/7b_mean_slerp_0305_emb.sh):  
```bash
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
        --transition_method slerp \
        --cache_dir "/path/to/cache" \
        --output_dir "/path/to/models/${model_abbrev}-${lang_code}-madlad-mean-slerp0305-emb"
        
done
```

### Linear
#### Merge

[`7b_mean_0305_emb.sh`](./scripts/qwen25/2x2ls/7b_mean_0305_emb.sh): 
```bash
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
```
