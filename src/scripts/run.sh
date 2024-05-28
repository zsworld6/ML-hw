#!/bin/bash

export PYTHONPATH="your_path_to/ML-hw:$PYTHONPATH"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python $DIR/main.py --model_path "your_path_to/ML-hw/models/Qwen-7B-Chat" \
    --data_path "your_path_to/ML-hw/Tasks/house-prices-advanced-regression-techniques" \
    --reflexion_method 1 \
    --reflexion_iterations 10 \
    --caafe_method 0 \
    --max_seq_len 2048 --max_batch_size 6 \
    --temperature 0.9 --top_k 1 --top_p 0.95 \
