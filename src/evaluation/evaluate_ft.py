import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..generation.prompts import gpt_conditioned_prompts_base
from ..utils.exllama_utils import ExLLamaModel
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


df, embds_dict = get_dataset(num_examples=5000)

model_0 = ExLLamaModel("/mnt/ssd/data/gen_prompt_results_0.65/exl2")
# model_0 = ExLLamaModel("/home/mpf/code/kaggle/llm-prompt/results/exl2")
# model_1 = ExLLamaModel("/mnt/ssd/data/Mistral-7B-Instruct-v0.2-exl2")

# model = ExLLamaModel("/mnt/ssd/data/Smaug-34B-v0.1-4.0bpw-h6-exl2", args={"length": 4096, "max_input_len": 4096})

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device="cpu")

scores = []
# line_break_token = model.tokenizer.encode("\n")[0][-1]

iter_num = 1


acc, acc_2 = 0.0, 0.0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = get_prompt_subject(row)
    # prompt = get_prompt_subject_quality(row)

    gen_text = model_0.generate(prompt, max_tokens=32)  # , stop_token=line_break_token)

    # success = gen_text.split("\n")[1].replace("Success: ", "").strip()
    gen_text = gen_text.split("\n")[0].strip()
    pred_prompt = gen_text
    
    # prompt_subject = get_ft_prompt_subject_2(gen_text)
    # pred_prompt = model_1.generate(prompt_subject, max_tokens=32)  # , stop_token=line_break_token)
    # pred_prompt = pred_prompt.replace('"', "").strip()
    # pred_prompt = pred_prompt.split("\n")[0].strip()
    # pred_prompt = pred_prompt[:-1] + ", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style."

    pred_prompt = f'Please improve the following text rewriting it with the subject: "{gen_text}", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'Please improve the following text rewriting it with the subject: , using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'Please improve the following text "{gen_text}" using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'{gen_text}. Use the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f"Improve the text to this {gen_text}."
    base_text = f"Improve the text to this." 
    # base_text = 'Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style.Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'

    # acc += row["rewrite_prompt"].split(" ")[0].lower() == pred_prompt.split(" ")[0].lower()
    # acc_2 += row["rewrite_prompt"].split(" ")[0].lower() == "rewrite"
    # print(acc / (idx+1), acc_2 / (idx+1), row["rewrite_prompt"].split(" ")[0].lower(), pred_prompt.split(" ")[0].lower())

    # if success.lower() == "no":
    #     pred_prompt = base_text

    score = get_embds_score(t5, pred_prompt, row["rewrite_prompt"])
    scores.append(score)

    # print(prompt)
    # print(len(prompt))
    print(gen_text)
    # print(success)
    # print(pred_prompt)
    print(row["rewrite_prompt"])
    print(score, "\n\n")
    print(np.mean(scores))
    # import pdb; pdb.set_trace()


scores = np.array(scores)

print(np.mean(scores))

import pdb

pdb.set_trace()


## 0.614

# NEW prompt: 0.5803
