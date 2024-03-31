import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .gpt_utils import call_gpt_api
from .prompts import gpt_conditioned_prompts_base


seed = 27007
np.random.seed(seed)


system_prompt = "You are tasked to assist into extracting the main idea of a rewrite prompt"

def build_fewshot_prompt(dct):
    prompt = \
f'''Prompt: {dct["prompt"]}
Subject: {dct["subject"]}
'''
    return prompt


def build_prompt(text_input, content):
    fewshot_prompt = "\n".join([build_fewshot_prompt(dct) for dct in content])
    prompt = f'''Given a rewrite prompt passed to a LLM, your task is to extract the subject of it.

I will provide some examples in the format:

Prompt: The prompt used in a rewrite task done by a LLM
Subject: The subject of the rewrite task, a couple of words that describe the main topic of the text

I will provide some examples followed by the input text for which you will complete the requested information as provided.

Your answer should consist only of the subject prediction and nothing else. I will parse it exactly as that.

Begin!

{fewshot_prompt}

Prompt: {text_input}
Subject: '''

    return prompt


def parse_result(result):
    result = result.replace("Subject:", "").strip()
    return result


def get_results(input_text, api_params={}):
    content = copy.deepcopy(gpt_conditioned_prompts_base)
    prompt = build_prompt(input_text, content)
    result = call_gpt_api(prompt, **api_params)

    try:
        result = parse_result(result)
    except:
        print("INVALID OUTPUT:\n", result, "\n")
        result = ""

    return result



input_path = Path("/kaggle/input/pedro-data/data.csv")
out_path = input_path.with_name("data_subject_3.csv")

dataset = pd.read_csv(input_path)
dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
# dataset = dataset.head(10000)
dataset = dataset.iloc[20000:30000].reset_index(drop=True)

api_params = {"system_prompt": system_prompt, "model": "gpt-3.5-turbo", "temperature": 0.1}
subjects = []

for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    subject = get_results(row["rewrite_prompt"], api_params=api_params)
    subjects.append(subject)

dataset["subject"] = subjects

dataset.to_csv(out_path, index=False)
