#!/bin/bash

cd /path/to/cva-merge/preprocessing/src/evaluation/

python create_mt_dataset.py \
    --data_dir \
    --output_dir /path/to/cva-merge/preprocessing \
    --cache_dir "/path/to/cache" \
    --repo_id your-hf-id/mt-${lang_code}
