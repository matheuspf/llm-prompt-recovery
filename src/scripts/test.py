import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
)

# model_name = "google/gemma-7b-it"
# model_name = "TechxGenus/gemma-7b-it-GPTQ"
# model_name = "TheBloke/CodeLlama-34B-Instruct-GPTQ"
# model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
# model_name = "TheBloke/Mistral-7B-v0.1-GPTQ"
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
token = open(os.path.expanduser("~/.hugginface_token")).read().strip()

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=False,
    # quantization_config=quantization_config,
    # low_cpu_mem_usage=True,
    # torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    token=token,
)

# input_text = "Write me a poem about Machine Learning."
# input_text = """<start_of_turn>user
# Write me a poem about Machine Learning<end_of_turn>
# <start_of_turn>model
# """

input_text = 'Convert this into a sea shanty: """"""The competition dataset comprises text passages that have been rewritten by the Gemma LLM according to some rewrite_prompt instruction. The goal of the competition is to determine what prompt was used to rewrite each original text.  Please note that this is a Code Competition. When your submission is scored, this example test data will be replaced with the full test set. Expect roughly 2,000 original texts in the test set."""""""""'

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")["input_ids"]


# chat = [{"role": "user", "content": input_text}]
# input_ids = tokenizer.apply_chat_template(
#     chat, add_generation_prompt=True, return_tensors="pt"
# ).to("cuda")

# print(tokenizer.batch_decode(input_ids))

start = time.time()

outputs = model.generate(input_ids=input_ids, max_new_tokens=128, temperature=0)

end = time.time()

result = tokenizer.decode(outputs[0])
len_gen_text = len(tokenizer.tokenize(result[len(input_text) :]))

print(result)

print(len_gen_text / (end - start))
