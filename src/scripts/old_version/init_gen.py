# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os, sys

# 获取外部文件的绝对路径
external_path = os.path.abspath('../../llama3')
print(external_path)

# 添加路径到 sys.path
if external_path not in sys.path:
    sys.path.append(external_path)

# 获取外部文件的绝对路径
external_path = os.path.abspath('./llama3')
print(external_path)

# 添加路径到 sys.path
if external_path not in sys.path:
    sys.path.append(external_path)


from typing import List, Optional

import fire

from llama import Dialog, Llama
import pandas as pd
    
external_path = os.path.abspath('./src/scripts')
print(external_path)

# 添加路径到 sys.path
if external_path not in sys.path:
    sys.path.append(external_path)


from utility import get_variable_from_file


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.95,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 2048,
    data_path = "",
    train_file = "",
    test_file = ""
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    
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

    dialogs: List[Dialog] = [[{"role": "system", "content": init_message}]]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for result in results:
        print(result['generation']['content'])


if __name__ == "__main__":
    fire.Fire(main)
