#!/bin/bash



DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GRANDPARENT_DIR="$(cd "$DIR/../.." && pwd)"
export PYTHONPATH="$GRANDPARENT_DIR/src:$PYTHONPATH"

python $DIR/main.py --model_path "$GRANDPARENT_DIR/models/Qwen-7B-Chat" \
    --data_path "$GRANDPARENT_DIR/Tasks/spaceship-titanic" \
    --prompt_method 1 \
    --reflexion_iterations 10 \
    --caafe_method 2 \
    --max_seq_len 2048 --max_batch_size 6 \
    --temperature 0.9 --top_k 1 --top_p 0.95 \
    --describe_task True \
