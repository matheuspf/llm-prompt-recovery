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
from Levenshtein import ratio


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


def get_df_train():
    data_list = [
        "/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated.csv",
        "/kaggle/input/pedro-data/data_subject.csv",
        "/kaggle/input/pedro-data/data_subject_2.csv",
        "/kaggle/input/pedro-data/data_subject_3.csv",
    ]
    
    
    # df = pd.read_csv("/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated.csv")
    # df = pd.read_csv("/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated_evaluated.csv")

    df = pd.concat([pd.read_csv(data) for data in data_list], ignore_index=True)

    return df


df, embds_dict = get_dataset(num_examples=500)
# df, embds_dict = get_dataset(num_examples=None)

# df_train = get_df_train()

# print(len(df))
# df = df[~df["original_text"].isin(df_train["original_text"].values.tolist())]
# df = df[~df["rewritten_text"].isin(df_train["rewritten_text"].values.tolist())]
# df = df[~df["rewrite_prompt"].isin(df_train["rewrite_prompt"].values.tolist())]
# print(len(df))


# model_0 = ExLLamaModel("/mnt/ssd/data/gen_prompt_results_0.65/exl2")
model_0 = ExLLamaModel("/home/mpf/code/kaggle/llm-prompt/results/exl2")
# model_1 = ExLLamaModel("/mnt/ssd/data/Mistral-7B-Instruct-v0.2-exl2")

# model = ExLLamaModel("/mnt/ssd/data/Smaug-34B-v0.1-4.0bpw-h6-exl2", args={"length": 4096, "max_input_len": 4096})

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device="cpu")

scores = []
# line_break_token = model.tokenizer.encode("\n")[0][-1]

iter_num = 1


for idx, row in tqdm(df.iterrows(), total=len(df)):
    # prompt = get_prompt_subject(row)
    gen_text = model_0.generate(prompt, max_tokens=32)  # , stop_token=line_break_token)
    # gen_text = gen_text.split("\n")[0].strip()

    prompt = get_prompt_json(row)
    gen_text = model_0.generate(prompt, max_tokens=32)
    gen_text = json_parser_from_chat_response(gen_text)["subject"]

    pred_prompt = gen_text
    
    # pred_prompt = f'Please improve the following text rewriting it with the subject: "{gen_text}", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'Retell text into this conveyed human piece if make such {gen_text}.'
    # pred_prompt = f'Improve rephrase text conveying portrayal tone throughout {gen_text}.'
    pred_prompt = f'Improve rephrase text manner this written to has language tone within to {gen_text}.'
    
    
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



# 0.6651