#!/bin/bash

cd /path/to/cva-merge/preprocessing/src/evaluation/

lang_codes=(
    "am"
    "bn"
    "gu"
    "my"
    "si"
    "ta"
    "te"
    "en"
)

for lang_code in "${lang_codes[@]}"; do
    python create_sum_dataset.py \
        --output_dir /path/to/cva-merge/preprocessing \
        --cache_dir "/path/to/cache" \
        --repo_id your-hf-id/sum-${lang_code} \
        --lang_code ${lang_code}
done
