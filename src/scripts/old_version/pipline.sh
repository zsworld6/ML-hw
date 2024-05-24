scripts_path="./src/scripts/old_version"
gen_path="./src/gen"
competition="house-prices-advanced-regression-techniques"
train_file="train.csv"
test_file="test.csv"

log_file="./src/temp/log"
output_file="./src/gen/gen_code_space_shit.py"
submission_file="./src/gen/submission.csv"
score_file="./src/temp/score"
model_path="./llama3/Meta-Llama-3-8B-Instruct/"
tokenizer_path="./llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"

torchrun --nproc_per_node 1 ./src/scripts/old_version/init_gen.py \
    --ckpt_dir $model_path \
    --tokenizer_path $tokenizer_path \
    --max_seq_len 4096 --max_batch_size 6 > $log_file \
    --data_path "./Tasks/$competition" \
    --train_file $train_file \
    --test_file $test_file


awk '/```/ {p=1;next} p' "$log_file" > "$output_file"

counter=0

while [ $counter -le 30 ]
do
    echo "iter $counter"

    error_file="./src/temp/error.log"
    is_error=0
    python "$output_file" 2> "$error_file"

    if [ $? -eq 0 ]; then
        echo "Generate a correct code"
        python ./src/kaggle/submit_kaggle.py --file_name $submission_file --competition $competition
        python ./src/kaggle/query_kaggle.py --competition $competition > $score_file
        echo "Score: $(cat $score_file)"
        cp $output_file "./src/gen/gen_code$counter.py"
        cp $submission_file "./src/gen/submission$counter.csv"
    else
        echo "Generate a wrong code"
        is_error=1
    fi

    torchrun --nproc_per_node 1 ./src/scripts/old_version/reflector.py \
        --ckpt_dir $model_path \
        --tokenizer_path $tokenizer_path \
        --max_seq_len 4096 \
        --max_batch_size 6 \
        --is_error $is_error > ./src/temp/reflect \
        --gen_code_file $output_file \
        --error_file $error_file \
        --score_file $score_file

    tail -n +5 ./src/temp/reflect > ./src/temp/reflections
    rm ./src/temp/reflect

    echo "Success to reflect"

    torchrun --nproc_per_node 1 ./src/scripts/old_version/actor.py \
        --ckpt_dir $model_path \
        --tokenizer_path $tokenizer_path \
        --max_seq_len 4096 --max_batch_size 6 > log \
        --gen_code_file $output_file \
        --reflections_file ./src/temp/reflections
    
    awk '/```/ {p=1;next} p' "$log_file" > "$output_file"

    echo "Success to generate new code"

    ((counter++))
done

python $output_file
