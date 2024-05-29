import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 设定设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据集
dataset = load_dataset("/home/shiqi/ML-hw/sft-data/AlpacaCode", split="train")

# 加载模型与分词器
model_name = "/home/shiqi/ML-hw/models/Qwen1___5-0___5B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA配置
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# 训练参数配置
training_arguments = TrainingArguments(
    output_dir="./Qwen1___5-0___5B-Chat-results1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=1000,
    logging_steps=100,
    learning_rate=2e-5,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=10000,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    group_by_length=True
)

# 初始化训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=2048,
    dataset_text_field="text"
)

# 训练模型
trainer.train()

# 保存模型
model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
model_to_save.save_pretrained("./Qwen1___5-0___5B-Chat-finetuned-model_code")
lora_config = LoraConfig.from_pretrained("./Qwen1___5-0___5B-Chat-finetuned-model_code")
model = get_peft_model(model, lora_config).to(device)

# 使用模型生成文本
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
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
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出生成的文本
print(response)
