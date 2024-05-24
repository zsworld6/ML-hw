from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from utility import get_variable_from_file
# 加载预训练的模型和配置
tokenizer = AutoTokenizer.from_pretrained("/home/shiqi/.cache/modelscope/hub/qwen/Qwen1___5-0___5B-Chat")
model = AutoModelForCausalLM.from_pretrained("/home/shiqi/.cache/modelscope/hub/qwen/Qwen1___5-0___5B-Chat")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载并应用 LoRA 配置
lora_config = LoraConfig.from_pretrained("/home/shiqi/src/data/finetuned-model_code")
model = get_peft_model(model, lora_config).to(device)
train_file="/home/shiqi/Tasks/house-prices-advanced-regression-techniques/train.csv"
test_file="/home/shiqi/Tasks/house-prices-advanced-regression-techniques/test.csv"
data_path = "/home/shiqi/Tasks/house-prices-advanced-regression-techniques"


example_columns = get_variable_from_file(f"{data_path}/columns.py", "example_columns")

predict_goal = get_variable_from_file(f"{data_path}/columns.py", "predict_goal")

init_message = f"""
You are a Python programming assistant. 
The dataframe '{data_path}/{train_file}' is loaded and in memory. Columns are also named attributes. 
Description of the dataset in '{train_file}':
```
{example_columns}
```
Columns dtypes are all unknown, and you should carefully think the dtype of each column by yourself.
Please choose the best way to implement a model to predict the {predict_goal}. '{data_path}/{train_file}' is the training data. '{data_path}/{test_file}' is the test data. And you should save your prediction in 'submission.csv'.
Please try your best to avoiding any possible errors when running your code!
You must **only** write a Python code for the implementation, without any other characters!
(Using RandomForest)
"""








# 使用模型生成文本
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": init_message}
]

# 将消息模板化并进行 tokenization
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 生成文本
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出生成的文本
print(response)
