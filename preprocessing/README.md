Preprocessing
===

**All the following scripts are written for the Slurm workload manager. You may need to modify the scripts to run them on your local machine.**

## Training
### 1. Generate tokenizer training data
First, you need to generate training data for the tokenizer. You can use the following script to generate training data for the tokenizer:

[`generate_tokenizer_training_data_madlad.sh`](./scripts/training/generate_tokenizer_training_data_madlad.sh):  
```bash
#!/bin/bash
#SBATCH --job-name=generate_tokenizer_training_data
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Configs
cd /path/to/chat-cve/preprocessing/src/training/
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
lang_code="$1"

# Run the script
output_file="/path/to/datasets/madlad-${lang_code}.txt"
python generate_tokenizer_training_data_madlad.py \
    --lang_code ${lang_code} \
    --output_file ${output_file} \
    --datasets_cache_dir ${HF_DATASETS_CACHE}
```

### 2. Train a tokenizer
Second, you need to train a tokenizer for each target language and source model. You can use the following script to train a tokenizer for each target language and source model:

#### Qwen2.5 and Llama 3.1
[`train_tokenizer_madlad.sh`](./scripts/training/train_tokenizer_madlad.sh):  
```bash
#!/bin/bash
#SBATCH --job-name=train_tokenizer_madlad
#SBATCH --mem=64G
#SBATCH --time=96:00:00

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
python train_tokenizer.py \
    --corpus_path ${corpus_path} \
    --vocab_size ${vocab_size} \
    --output_dir ${output_dir} \
    --lang_code ${lang_code} \
    --num_new_tokens ${num_new_tokens} \
    --datasets_cache_dir "${HF_DATASETS_CACHE}" \
    --hub_cache_dir "${HF_HUB_CACHE}" \
    --model_name_or_path ${model_name_or_path}
```

#### Gemma 2
[`train_tokenizer_gemma2_sp_madlad.sh`](./scripts/training/train_tokenizer_gemma2_sp_madlad.sh):  
```bash
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
```

### 3. Tokenize the dataset
Third, you need to tokenize the dataset for each target language and source model. You can use the following script to tokenize the dataset for each target language and source model:

#### Vocabulary Expansion
[`generate_lapt_data_madlad.sh`](./scripts/training/generate_lapt_data_madlad.sh):  
```bash
#!/bin/bash
#SBATCH --job-name=generate_lapt_data_madlad
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

# Run the script
output_dir="/path/to/datasets/${model_abbrev}-${lang_code}-madlad/"
tokenizer_name_or_path="/path/to/tokenizers/${model_abbrev}-${lang_code}-madlad/"

python generate_lapt_data_madlad.py \
    --lang_code ${lang_code} \
    --output_dir ${output_dir} \
    --datasets_cache_dir ${HF_DATASETS_CACHE} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --tokenizer_cache_dir ${HF_HUB_CACHE} \
    --num_workers 4 \
    --max_length 512
```

#### CPT-only
[`generate_lapt_data_madlad_src.sh`](./scripts/training/generate_lapt_data_madlad_src.sh):  
```bash
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
```

