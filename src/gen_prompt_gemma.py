import time

import numpy as np
import pandas as pd
import torch
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .data import *
from .exllama_utils import ExLLamaModel, Perplexity, get_gemma_prompt

np.random.seed(42)


def build_prompt(prompt, original_text):
    prompt_ = f'''<start_of_turn>user
{prompt}
""""""
{original_text}
""""""<end_of_turn>
<start_of_turn>model
'''
    return prompt_


data_org = load_data()

# data_key = "gemma_gen"
data_key = "nbroad"
data = data_org[data_key]

prompt_df = get_gen_prompts()
prompts_list = prompt_df["rewrite_prompt"].tolist()


original_text_list = []

for text in data["original_text"]:
    text_list = text.strip().split("\n")
    idx = 0
    
    while idx < len(text_list) and len("\n".join(text_list[idx])) < 400:
        idx += 1
    idx += 1
    
    text = "\n".join(text_list[:idx]).strip()

    if len(text) > 300 and len(text) < 500:
        original_text_list.append(text)

print(len(original_text_list), len(data))


# output_path = Path("/kaggle/input/gemma_7b_gen")
output_path = Path("/kaggle/input/gemma_7b_gen_v2")
output_path.mkdir(parents=True, exist_ok=True)

original_text = []
rewritten_text = []


with torch.inference_mode():
    model = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2")

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.1
    settings.top_k = 1
    settings.top_p = 0.75
    settings.token_repetition_penalty = 1.00
    # settings.disallow_tokens(model.tokenizer, [model.tokenizer.eos_token_id])

    generator = ExLlamaV2BaseGenerator(model.model, model.cache, model.tokenizer)
    generator.warmup()

    for idx, prompt in enumerate(tqdm(prompts_list)):
        text = original_text_list[idx % len(original_text_list)]

        ids = model.tokenizer.encode(prompt)
        tokens_prompt = ids.shape[-1]

        time_begin = time.time()

        prompt = build_prompt(prompt, text)

        output = generator.generate_simple(
            prompt, settings, num_tokens=2048, token_healing=True, add_bos=True
        )

        torch.cuda.synchronize()

        time_end = time.time()

        gen_text = output.replace(prompt, "").strip() 

        if gen_text[:4].lower() == "sure":
            gen_text = gen_text.split("\n", 1)[1].strip()
        
        original_text.append(text)
        rewritten_text.append(gen_text)

        print(f"Time: {time_end - time_begin:.2f}s")
        print("\n\n")
        # print(gen_text)
        print(output)
        print("\n\n")


prompt_df["original_text"] = original_text
prompt_df["rewritten_text"] = rewritten_text

prompt_df.to_csv(output_path / "gemma_7b_exl2_gen.csv", index=False)
