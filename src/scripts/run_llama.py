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

from typing import List, Optional, Dict

import fire

from llama import Dialog, Llama
import pandas as pd

from reflexion import run_reflexion

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 2048,
    data_path = "",
    train_file = "",
    test_file = ""
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    def chat(messages: List[Dict]) -> str:
        dialogs: List[Dialog] = [
            [{"role": message["role"], "content": message["content"]} for message in messages]
        ]
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return results[0]['generation']['content']
    
    run_reflexion(chat, 100, data_path)
    

if __name__ == "__main__":
    fire.Fire(main)
