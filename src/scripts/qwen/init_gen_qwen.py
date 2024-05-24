from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from utility import get_variable_from_file
device = "cuda" if torch.cuda.is_available() else "cpu"
train_file="/home/shiqi/Tasks/house-prices-advanced-regression-techniques/train.csv"
test_file="/home/shiqi/Tasks/house-prices-advanced-regression-techniques/test.csv"
model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen1.5-0.5B-Chat",
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-0.5B-Chat")


# #########################
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
"""

# dialogs: List[Dialog] = [[{"role": "system", "content": init_message}]]
#############################
# prompt = "You are a Python programming assistant. "
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": init_message}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
