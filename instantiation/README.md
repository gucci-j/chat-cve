Initialize Vocabulary Expanded Model
===

Note: The following scripts are for initializing the vocabulary expanded model using the `mean` method. You need to replace all the paths in the scripts with the correct paths.

## Qwen2.5
### Chat
You can initialize the vocabulary expanded model using the `Qwen/Qwen2.5-7B-Instruct` model with the following script. The script will initialize the vocabulary expanded model for the languages `am`, `bn`, `gu`, `my`, `si`, `ta`, and `te`.

[`7b_mean_sft.sh`](./scripts/qwen25/7b_mean_sft.sh):
```bash
#!/bin/bash
#SBATCH --job-name=instantiate_qwen25_sft_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Configs
cd /path/to/cva-merge/instantiation/src
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
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
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

```

### Base
You can initialize the vocabulary expanded model using the `Qwen/Qwen2.5-7B` model with the following script. The script will initialize the vocabulary expanded model for the languages `am`, `bn`, `gu`, `my`, `si`, `ta`, and `te`.

[`7b_mean.sh`](./scripts/qwen25/7b_mean.sh):
```bash
#!/bin/bash
#SBATCH --job-name=instantiate_qwen25_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Configs
cd /path/to/cva-merge/instantiation/src
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
model_name_or_path="Qwen/Qwen2.5-7B"
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

```

## Llama 3.1
### Chat
You can initialize the vocabulary expanded model using the `meta-llama/Llama-3.1-8B-Instruct` model with the following script. The script will initialize the vocabulary expanded model for the languages `am`, `bn`, `gu`, `my`, `si`, `ta`, and `te`.

[`8b_mean_sft.sh`](./scripts/llama31/8b_mean_sft.sh):
```bash
#!/bin/bash
#SBATCH --job-name=instantiate_llama31_8b_sft_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00


# Configs
cd /path/to/cva-merge/instantiation/src
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
model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)

# Run the script
for lang_code in "${lang_codes[@]}"; do
    tokenizer_dir="/path/to/tokenizers/Llama-3.1-8B-${lang_code}-madlad/"
    output_dir="/path/to/models/${model_abbrev}-${lang_code}-madlad-mean/"
    
    python main.py \
        --source_model_name_or_path ${model_name_or_path} \
        --target_tokenizer_name_or_path ${tokenizer_dir} \
        --output_dir ${output_dir} \
        --cache_dir ${HF_HUB_CACHE} \
        --method mean

done

```

### Base

You can initialize the vocabulary expanded model using the `meta-llama/Llama-3.1-8B` model with the following script. The script will initialize the vocabulary expanded model for the languages `am`, `bn`, `gu`, `my`, `si`, `ta`, and `te`.

[`8b_mean.sh`](./scripts/llama31/8b_mean.sh):

```bash
#!/bin/bash
#SBATCH --job-name=instantiate_llama31_8b_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Configs
cd /path/to/cva-merge/instantiation/src
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
model_name_or_path="meta-llama/Llama-3.1-8B"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)

# Run the script
for lang_code in "${lang_codes[@]}"; do
    tokenizer_dir="/path/to/tokenizers/Llama-3.1-8B-${lang_code}-madlad/"
    output_dir="/path/to/models/${model_abbrev}-${lang_code}-madlad-mean/"
    
    python main.py \
        --source_model_name_or_path ${model_name_or_path} \
        --target_tokenizer_name_or_path ${tokenizer_dir} \
        --output_dir ${output_dir} \
        --cache_dir ${HF_HUB_CACHE} \
        --method mean

done

```

## Gemma 2
### Chat
You can initialize the vocabulary expanded model using the `google/gemma-2-9b-it` model. The following script will initialize the vocabulary expanded model for the languages `am`, `bn`, `gu`, `my`, `si`, `ta`, and `te`.

[`9b_mean_sft.sh`](./scripts/gemma2/9b_mean_sft.sh):
```bash
#!/bin/bash
#SBATCH --job-name=instantiate_gemma2_sft_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Configs
cd /path/to/cva-merge/instantiation/src
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

```

### Base
You can initialize the vocabulary expanded model using the `google/gemma-2-9b` model. The following script will initialize the vocabulary expanded model for the languages `am`, `bn`, `gu`, `my`, `si`, `ta`, and `te`.

[`9b_mean.sh`](./scripts/gemma2/9b_mean.sh):

```bash
#!/bin/bash
#SBATCH --job-name=instantiate_gemma2_mean
#SBATCH --mem=64G
#SBATCH --time=24:00:00


# Configs
cd /path/to/cva-merge/instantiation/src
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
model_name_or_path="google/gemma-2-9b"
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

```
