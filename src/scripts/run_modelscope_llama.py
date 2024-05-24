import transformers
import torch
from modelscope import snapshot_download
from typing import List, Optional, Dict
import fire
from reflexion import run_reflexion


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 2048,
    data_path = "",
    train_file = "",
    test_file = ""
):
    model_id = snapshot_download("LLM-Research/Meta-Llama-3-8B-Instruct")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    
    def chat(messages: List[Dict]) -> str:
        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        tokenizer = pipeline.tokenizer
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=max_gen_len,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0]["generated_text"][len(prompt):]
    
    run_reflexion(chat, 10, data_path)
    

if __name__ == "__main__":
    fire.Fire(main)