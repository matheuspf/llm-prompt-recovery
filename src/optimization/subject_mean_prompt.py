import os
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..generation.prompts import gpt_conditioned_prompts_base
from ..utils.exllama_utils import ExLLamaModel
from ..evaluation.prompts import *
from Levenshtein import ratio
from .data import *



def get_mean_prompt():
    # return "Rewrite this text convey manner human evokes text better exude genre plath tone cut include object being about please further wise this individuals could originally convey here."
    return "Rewrite this lucrarea tone text conveyimprove lucrareaENCE prompt / text wiseOf lucrareaEleDearrepede].phraseol stuff appealingOF a bourneggcontemporainR."


def eval_mean_prompt(model, t5, mean_prompt):
    scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = get_prompt_subject(row)
        subject = model.generate(prompt, max_tokens=32)
        subject = subject.split("\n")[0].strip()
        # subject = subject.split(",")[0].strip()

        # subject = ""
        
        pred_prompt = mean_prompt.replace("{{subject}}", subject)

        score = get_embds_score(t5, pred_prompt, row["rewrite_prompt"])
        scores.append(score)

        print(subject)
        print(row["rewrite_prompt"])
        print(score, "\n\n")
        print(np.mean(scores))

    scores = np.array(scores)
    mean_score = np.mean(scores)

    return mean_score



# df = get_dataset_pub()
# df_gpt = get_dataset_gpt()
# df_prompts = get_dataset_pedro()

# print(len(df))
# df = df[df["rewrite_prompt"].isin(set(df_prompts["rewrite_prompt"].values))].reset_index(drop=True)
# df = df[~df["rewrite_prompt"].isin(set(df_gpt["rewrite_prompt"].values))].reset_index(drop=True)
# print(len(df))

# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# df = df.head(100)



df = get_gen_sel_dataset()
df_gpt = get_dataset_gpt()
print(len(df))
df = df[~df["rewrite_prompt"].isin(set(df_gpt["rewrite_prompt"].values))].reset_index(drop=True)
print(len(df))
df = df.head(100)


mean_prompt = get_mean_prompt()
mean_prompt_list = mean_prompt.split(" ")
print(len(mean_prompt_list), mean_prompt_list)


model = ExLLamaModel("/mnt/ssd/data/gen_prompt_results_0.65/exl2")

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device="cuda:0")

mean_scores = {}

# for pos in tqdm(range(1, len(mean_prompt_list) + 1)):
for pos in tqdm(range(len(mean_prompt_list) + 1, len(mean_prompt_list) + 2)):
    cur_mean_prompt_list = copy.deepcopy(mean_prompt_list)
    cur_mean_prompt_list.insert(pos, '{{subject}}')
    # cur_mean_prompt_list.insert(pos, '"{{subject}}"')
    cur_mean_prompt = " ".join(cur_mean_prompt_list)

    mean_score = eval_mean_prompt(model, t5, cur_mean_prompt)
    mean_scores[pos] = float(mean_score)

    print("\n")
    print(pos, mean_score)
    print(cur_mean_prompt)
    print("\n\n")

with open("subject_mean_prompt_scores_first_sentence.json", "w") as f:
    json.dump(mean_scores, f, indent=4)
