import os
import torch
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..generation.prompts import gpt_conditioned_prompts_base
from ..utils.exllama_utils import ExLLamaModel, Perplexity
from .prompts import *


def get_embds_score(t5, pred, gt):
    pred_embds = t5.encode(pred, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    gt_embds = t5.encode(gt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)

    res = abs((cosine_similarity(gt_embds, pred_embds)) ** 3)

    return res[0][0]


def get_dataset(data_path="/kaggle/input/df_with_emb.parquet", num_examples=None):
    parquet = pd.read_parquet(data_path)

    if num_examples is not None:
        # parquet = parquet.head(num_examples)
        parquet = parquet.sample(num_examples, random_state=42)

    embds_dict = {}

    for c in ("original_text", "rewrite_prompt", "rewritten_text"):
        embds_dict[c] = np.array(parquet[[f"{c}_emb_{i}" for i in range(768)]].values.tolist())

    df = parquet[["original_text", "rewrite_prompt", "rewritten_text"]].reset_index(drop=True)

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(
        drop=True
    )
    df = df[df["rewritten_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(
        drop=True
    )

    return df, embds_dict


def build_gemma_prompt(prompt, original_text, rewritten_text):
    prompt_ = f"""<start_of_turn>user
{prompt}
{original_text}<end_of_turn>
<start_of_turn>model
{rewritten_text}<end_of_turn>"""
    return prompt_


def get_best_perplexity(model, prompts, original_text, rewritten_text):
    perp_fn = Perplexity()
    perps = []

    model_start_token = model.tokenizer.encode("<start_of_turn>")[0]
    bos_num_tokens = 3 # <bos><start_of_turn>user

    for prompt in prompts:
        gemma_prompt = build_gemma_prompt(prompt, original_text, rewritten_text)
        
        with torch.inference_mode():
            ids = model.tokenizer.encode(gemma_prompt, encode_special_tokens=True, return_offsets=False, add_bos=True)
            answer_start = (ids[0] == model_start_token)[bos_num_tokens:].long().argmax().item() + bos_num_tokens + 3

            model.cache.current_seq_len = 0
            model.model.forward(ids[:, :answer_start], model.cache, preprocess_only=True)
            output = model.model.forward(ids[:, answer_start:], model.cache).float().cpu()
            
            labels = ids
            output_answer = output[:, :-1]
            labels_answer = labels[:, answer_start:-1]

            p = perp_fn(output_answer, labels_answer).item()
            perps.append(p)

    perps = np.array(perps)
    best_perp_idx = np.argmin(perps)
    best_prompt = prompts[best_perp_idx]

    return best_prompt




df, embds_dict = get_dataset(num_examples=5000)

mistral = ExLLamaModel("/mnt/ssd/data/gen_prompt_results_0.65/exl2", args={"length": 2048, "max_input_len": 2048})
gemma = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2", args={"length": 2048, "max_input_len": 2048})
t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device="cpu")
perp = Perplexity()

scores = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = get_prompt_subject(row)

    gen_text = mistral.generate(prompt, max_tokens=32)  # , stop_token=line_break_token)
    gen_text = gen_text.split("\n")[0].strip()

    pred_prompt = f'Please improve the following text rewriting it with the subject: "{gen_text}", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    base_prompt = 'Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style.Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'

    prompts = [base_prompt, pred_prompt]
    best_prompt = get_best_perplexity(gemma, prompts, row["original_text"], row["rewritten_text"])
    
    
    score = get_embds_score(t5, best_prompt, row["rewrite_prompt"])
    scores.append(score)

    print(gen_text)
    print(gen_text in pred_prompt)
    print(best_prompt)
    print(row["rewrite_prompt"])
    print(score, "\n\n")
    print(np.mean(scores))


scores = np.array(scores)

print(np.mean(scores))

import pdb

pdb.set_trace()


## 0.614

# NEW prompt: 0.5803
