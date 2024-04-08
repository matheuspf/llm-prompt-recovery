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
from itertools import permutations, chain

def get_permutations(x, max_length):
    return list(chain.from_iterable(permutations(x, r) for r in range(1, max_length + 1)))


def clean_text(text):
    return "".join([t for t in text if t.isalpha() or t in (" ",)]).lower()
    # return text


def find_optimal_subject(t5, mean_prompt, df, max_words=3, max_test = 1000):
    best_subjects = []

    mean_prompt = clean_text(mean_prompt) + " {{subject}}"
    scores_list = []
    
    # for idx, row in tqdm(df.iterrows(), total=len(df)):
    for idx, row in df.iterrows():
        prompt = clean_text(row["rewrite_prompt"])

        prompt_words = prompt.split(" ")
        prompt_words = get_permutations(prompt_words, max_words)
        prompt_words = sorted(prompt_words, key=lambda x: len(x), reverse=True)
        prompt_words = prompt_words[:max_test]

        prompts = [clean_text(mean_prompt.replace("{{subject}}", " ".join(words))) for words in prompt_words]
        embds = t5.encode(prompts + [prompt], normalize_embeddings=True, show_progress_bar=False, batch_size=64)

        scores = (cosine_similarity(embds[:-1], embds[-1].reshape(1, -1)) ** 3)

        best_words = prompt_words[np.argmax(scores)]
        best_subject = " ".join(best_words)

        print(prompt[:50] + " ...", " | ", best_subject, " | ", np.max(scores))

        scores_list.append(np.max(scores))
        best_subjects.append(best_subject) 

        print(np.mean(scores_list))
    
    df["best_subject"] = best_subjects

    return df 
        
        

def get_mean_prompt():
    # return "Rewrite this text convey manner human evokes text better exude genre plath tone cut include object being about please further wise this individuals could originally convey here."
    # return "conveying rephraselucrarea textimprovelucrarealucrarea formal paragraph help please creativelywstlucrarea tonealterations ence text comportthislucrarea messageresemblepoeticallylucrarea casuallyoper talkingpresentingstoryinvolvesmemo essrecommendtransformingthisdetailsresponsivephrasethr reframe esstagline writerell it"
    # return "rephrase text better lucrarea lucrarea tone style discours involving a lucrarea creatively adv detail write this emulate casually sender lucrarea srl recompose a text contents"

    return "improve phrasing text lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea"


def eval_mean_prompt(model, t5, mean_prompt):
    scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = get_prompt_subject(row)
        subject = model.generate(prompt, max_tokens=32)
        subject = subject.split("\n")[0].strip()
        # subject = subject.split(",")[0].strip()

        # subject = ""
        
        pred_prompt = mean_prompt.replace("{{subject}}", subject)

        # score = get_embds_score(t5, pred_prompt, row["rewrite_prompt"])
        score = get_embds_score(t5, clean_text(pred_prompt), clean_text(row["rewrite_prompt"]))
        scores.append(score)

        # print(subject)
        # print(row["rewrite_prompt"])
        # print(score, "\n\n")
        # print(np.mean(scores))

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



# df = get_dataset_pedro()

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

# df = find_optimal_subject(t5, get_mean_prompt(), df, max_words=3)

mean_scores = {}

for pos in tqdm(range(0, len(mean_prompt_list) + 1)):
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


