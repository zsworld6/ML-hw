from typing import List, Union, Optional, Literal
import dataclasses
import transformers
import torch
import argparse

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

class ModelBase():
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate(self, messages: List[Message], args) -> Union[List[str], str]:
        raise NotImplementedError

class llama3Model(ModelBase):
    def __init__(self, path: str):
        super().__init__("llama3", path)
        from modelscope import snapshot_download
        # model_id = snapshot_download(path)
        pipeline = transformers.pipeline(
            "text-generation",
            model=path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.pipeline = pipeline
        self.terminators = terminators
    
    def generate(self, messages: List[Message], args) -> List[str] | str:
        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=args.max_seq_len,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        return outputs[0]["generated_text"][len(prompt):]
    
class llama3_4bitModel(ModelBase):
    def __init__(self, name:str, path: str):
        super().__init__(name, path)
        self.path = path
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        
    def generate(self, messages: List[Message], args) -> List[str] | str:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)  # 启用原生的2倍快速推理
        prompt = "\n".join([message["content"] for message in messages])
        prompt = "\n".join([prompt, "[START]"])
        print("----------------------------------------------")
        print(prompt)
        print("----------------------------------------------")        
        inputs = self.tokenizer(
        [
            prompt
        ], return_tensors = "pt").to("cuda")                        
        outputs = self.model.generate(**inputs, max_new_tokens = args.max_seq_len, use_cache = True)        
        generated_text = self.tokenizer.batch_decode(outputs)[0]        
        print(generated_text)
        start_token = "[START]"        
        start_index = generated_text.find(start_token) + len(start_token)
        if start_index == -1:
            start_index = 0
        response_text = generated_text[start_index:].strip()
        return response_text
    
class llama3_4bitModel_unfted(llama3_4bitModel):
    def __init__(self, path: str, max_seq_length: int):
        super().__init__("llama3_4bit_unfted", path)
        self.path = path
        from unsloth import FastLanguageModel        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.path,
            max_seq_length=max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )        
        
    def generate(self, messages: List[Message], args) -> List[str] | str:
        return super().generate(messages, args)
    
class llama3_4bitModel_sfted(llama3_4bitModel):
    def __init__(self, path: str, max_seq_length: int):
        super().__init__("llama3_4bit_unfted", path)
        self.path = path
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.path,  # 你用于训练的模型
            max_seq_length=max_seq_length,  # max_seq_length 应该在这段代码之前定义
            dtype=self.dtype,  # dtype 应该在这段代码之前定义
            load_in_4bit=self.load_in_4bit,  # load_in_4bit 应该在这段代码之前定义
        )
        FastLanguageModel.for_inference(self.model)  # 启用原生的2倍快速推理
        
    def generate(self, messages: List[Message], args) -> List[str] | str:
        return super().generate(messages, args)
    
class qwenModel(ModelBase):
    def __init__(self, path: str):
        super().__init__("qwen", path)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(self, messages: List[Message], args) -> List[str] | str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.max_seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=int(args.top_k),
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
def get_model_from_path(path, max_seq_length = 1024):
    if "4bit" in path:
        if "unfted" in path:
            return llama3_4bitModel_unfted(path, max_seq_length)        
        elif "sfted" in path or "fted" in path or "ftned" in path:
            return llama3_4bitModel_sfted(path, max_seq_length)
        else:
            return llama3_4bitModel_unfted(path, max_seq_length)
    elif "llama" in path or "Llama" in path:
        return llama3Model(path)
    elif "qwen" in path or "Qwen" in path:
        return qwenModel(path)
    else:
        raise ValueError(f"Unknown model path: {path}")