from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

from typing import List, Dict
from reflexion import run_reflexion

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen1.5-0.5B-Chat",
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-0.5B-Chat")

def chat(messages: List[Dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=4096,
        temperature=0.1
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

data_path = './Tasks/house-prices-advanced-regression-techniques'
run_reflexion(chat, 6, data_path)
