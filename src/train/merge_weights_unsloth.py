import torch
from unsloth import FastLanguageModel

# weights_path = "mistralai/Mistral-7B-Instruct-v0.2"
weights_path = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
loras_path = "./results/checkpoint-1200"
out_path = "./results/merged"

model, tokenizer = FastLanguageModel.from_pretrained(
    loras_path,
    load_in_4bit=True,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
FastLanguageModel.for_inference(model)

model.save_pretrained_merged(out_path, tokenizer, save_method="merged_16bit")
