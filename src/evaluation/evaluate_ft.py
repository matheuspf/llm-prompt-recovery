from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.exllama_utils import ExLLamaModel
from sentence_transformers import SentenceTransformer
from ..generation.prompts import gpt_conditioned_prompts_base


def get_prompt(dct):
        prompt = \
f'''[INST] Given an original text and the rewritten version of it from a LLM, predict what was the subject of the used prompt for the rewrite.

Original text:
""""""
{dct["original_text"]}
""""""

Rewritten text:
""""""
{dct["rewritten_text"]}
""""""

Subject:
""""""
[/INST] 
'''
        return prompt


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

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(drop=True)
    df = df[df["rewritten_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(drop=True)

    return df, embds_dict



df, embds_dict = get_dataset(num_examples=5000)

model = ExLLamaModel("/mnt/ssd/data/Mistral-7B-Instruct-v0.2-exl2")
model = ExLLamaModel("/home/mpf/code/kaggle/llm-prompt/results/exl2")

# model = ExLLamaModel("/mnt/ssd/data/Smaug-34B-v0.1-4.0bpw-h6-exl2", args={"length": 4096, "max_input_len": 4096})

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base")

scores = []
line_break_token = model.tokenizer.encode("\n")[0][-1]

iter_num = 1


for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = get_prompt(row)

    final_gen_text = ""
    for iter_idx in range(iter_num):
        if iter_idx > 0:
            final_gen_text += ", "
            prompt += ", "
            
        gen_text = model.generate(prompt, max_tokens=16)#, stop_token=line_break_token)
        gen_text = gen_text.replace('"', '').strip()
        gen_text = gen_text.split("\n")[0].strip()

        final_gen_text += gen_text
        prompt += gen_text
    
    gen_text = final_gen_text

    pred_prompt = f'Please improve the following text rewriting it with the subject: "{gen_text}", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'Please improve the following text "{gen_text}" using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'{gen_text}. Use the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'

    score = get_embds_score(t5, pred_prompt, row["rewrite_prompt"])
    scores.append(score)

    # print(prompt)
    # print(len(prompt))
    print(gen_text)
    print(row["rewrite_prompt"])
    print(score, "\n\n")
    print(np.mean(scores))
    # import pdb; pdb.set_trace()


scores = np.array(scores)

print(np.mean(scores))

import pdb; pdb.set_trace()


## 0.614

# NEW prompt: 0.5803

