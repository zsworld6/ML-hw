#!/bin/bash



DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GRANDPARENT_DIR="$(cd "$DIR/../.." && pwd)"
export PYTHONPATH="$GRANDPARENT_DIR/src:$PYTHONPATH"

python $DIR/main.py --model_path "$GRANDPARENT_DIR/models/Meta-Llama-3-8B-Instruct" \
    --data_path "$GRANDPARENT_DIR/Tasks/Mobile_Price_Classification" \
    --reflexion_method 1 \
    --reflexion_iterations 10 \
    --caafe_method 0 \
    --max_seq_len 2048 --max_batch_size 6 \
    --temperature 0.9 --top_k 1 --top_p 0.95 \
