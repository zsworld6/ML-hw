# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire, os, sys

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

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 2048,
    is_error = False,
    gen_code_file = "",
    error_file = "",
    score_file = ""    
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

    f_code = open(gen_code_file)
    code = f_code.read()
    

    if is_error:
        f_err = open(error_file)
        error_message = f_err.read()
        user_message = f"""
You are a helpful Python programming assistant. 
You will be given your previous implemention of a predictor. And you should self-reflect your previous implemention.
An error happened when running, so you should try your best to avoiding errors when running your code!
Your previous code:
```
{code}
```
And this is the error message:
```
{error_message}
```
Give me a suggestion of how to modify the code.
Please try your best to give me helpful suggestions! It's very important to me.
"""
    else:
        f_score = open(score_file)
        score = f_score.read()
        if score:
            user_message = f"""
You are a helpful Python programming assistant. 
You will be given your previous implemention of a predictor. And you should self-reflect your previous implemention.
The error rate of your predictor is {score}. Apply the necessary changes below by responding only with the improved body of the code.
Your previous code:
```
{code}
```
Please try your best to give me helpful suggestions! It's very important to me.
    """
        else:
            user_message = f"""
You are a helpful Python programming assistant. 
You will be given your previous implemention of a predictor. And you should self-reflect your previous implemention.
The dtypes of some columns of submission.csv may be wrong. Apply the necessary changes below by responding only with the improved body of the code.
Your previous code:
```
{code}
```
Please try your best to give me helpful suggestions! It's very important to me.
    """

    dialogs: List[Dialog] = [[{"role": "user", "content": user_message}]]
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
