export MODEL_NAME="llama-3-8b-Instruct-bnb-4bit_sft"
export DATA_DIR="/home/shiqi/ML-hw/models/LLaMA-Factory/data"
export DATA_NAME="codealpaca"
export BASE_MODEL="/home/shiqi/ML-hw/models/llama-3-8b-Instruct-bnb-4bit" # JUST AN EXAMPLE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cp finetune/dataset_info.json LLaMA-Factory/data/
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $DIR/../../models/LLaMA-Factory

python \
    src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${BASE_MODEL} \
    --finetuning_type lora \
    --template "default" \
    --dataset_dir ${DATA_DIR} \
    --dataset ${DATA_NAME} \
    --cutoff_len 4096 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --preprocessing_num_workers 8 \
    --max_steps 1500 \
    --save_steps 500 \
    --warmup_steps 100 \
    --output_dir checkpoints/${MODEL_NAME} \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir \
    # --resume_from_checkpoint checkpoints/${MODEL_NAME}/checkpoint-1000