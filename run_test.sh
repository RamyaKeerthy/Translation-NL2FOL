#!/bin/bash

python run_seq2seq.py \
    --model_name_or_path /home/ \
    --do_predict \
    --test_file data/ \
    --text_column input \
    --source_prefix "" \
    --output_dir output/ \
    --cache_dir  \
    --per_device_eval_batch_size=16 \
    --predict_with_generate
