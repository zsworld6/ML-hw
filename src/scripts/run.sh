#!/bin/bash

torchrun --nproc_per_node 1 src/scripts/run_modelscope_llama.py \
    --ckpt_dir llama3/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path llama3/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --data_path Tasks/house-prices-advanced-regression-techniques \
    --max_seq_len 4096 --max_batch_size 6
