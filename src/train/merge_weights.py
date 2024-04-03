import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

weights_path = "mistralai/Mistral-7B-Instruct-v0.2"
# loras_path = "/mnt/ssd/data/gen_prompt_results_0.65/checkpoint-5000"
# loras_path = "./results/finetuned_mistral"
loras_path = "./results/checkpoint-5000"
out_path = "./results/merged"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    weights_path,
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)


model = PeftModel.from_pretrained(model, loras_path)
model = model.merge_and_unload()

# tokenizer = AutoTokenizer.from_pretrained(weights_path)
tokenizer = AutoTokenizer.from_pretrained(loras_path)

model.save_pretrained(out_path)
tokenizer.save_pretrained(out_path)
