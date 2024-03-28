import json
import pandas as pd
from tqdm import tqdm
import copy
import numpy as np
import time
from pathlib import Path
from datasets import load_dataset
from .gpt_utils import call_gpt_api
from .prompts import gpt_conditioned_prompts_base


np.random.seed(27003)


def filter_data(text_list, min_len=300, max_len=3000, max_size=100):
    text_list = [text for text in text_list if len(text) >= min_len and len(text) <= max_len]

    np.random.shuffle(text_list)
    text_list = text_list[:max_size]
    
    return text_list


def text_head(text, min_len=300, split="\n"):
    text_list = text.split(split)[1:]
    idx = 1

    while idx < len(text_list) and len(split.join(text_list[:idx])) < min_len:
        idx += 1
    
    text = split.join(text_list[:idx]).strip()

    return text


def load_datasets(min_len=300, max_len=3000, max_size=100):
    text_list = []
    
    # https://huggingface.co/datasets/wikipedia
    wikipedia = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
    wiki_text = [dct["text"] for dct in wikipedia["train"]]
    # wiki_text = [f'{dct["title"]}\n{dct["text"]}' for dct in wikipedia["train"]]
    wiki_text = [text_head(text, min_len=2*min_len) for text in wiki_text]
 
    
    # https://huggingface.co/datasets/multi_news
    multi_news = load_dataset("multi_news", trust_remote_code=True)
    news_text = [dct["summary"].replace("– ", "") for dct in multi_news["train"]]


    # https://huggingface.co/datasets/webis/tldr-17
    reddit = load_dataset("webis/tldr-17", trust_remote_code=True)
    reddit_text = [dct["content"] for dct in reddit["train"]]

    
    # https://huggingface.co/datasets/imdb
    imdb = load_dataset("imdb", trust_remote_code=True)
    imdb_text = [dct["text"].replace("<br /><br />", "\n") for dct in imdb["train"]]


    # Merge
    wiki_text = filter_data(wiki_text, min_len=min_len, max_len=max_len, max_size=max_size)
    news_text = filter_data(news_text, min_len=min_len, max_len=max_len, max_size=max_size)
    reddit_text = filter_data(reddit_text, min_len=min_len, max_len=max_len, max_size=max_size)
    imdb_text = filter_data(imdb_text, min_len=min_len, max_len=max_len, max_size=max_size)

    text_list = wiki_text + news_text + reddit_text + imdb_text
    dataset_id = ["wikipedia"] * len(wiki_text) + ["multi_news"] * len(news_text) + ["reddit"] * len(reddit_text) + ["imdb"] * len(imdb_text)

    data = pd.DataFrame({
        "original_text": text_list,
        "dataset_id": dataset_id
    })

    return data



system_prompt = "You are tasked to design text rewrite prompts for a LLM"


def build_fewshot_prompt(dct):
    prompt = \
f'''### Input Text:
""""""
{dct["original_text"]}
""""""

### Info
Type: {dct["type"]}
Prompt: {dct["prompt"]}
Subject: {dct["subject"]}

'''
    return prompt


def build_prompt(input_text, content):
    fewshot_prompt = "\n".join([build_fewshot_prompt(dct) for dct in content])
    prompt = \
f'''Given an `Input Text`, you will answer:

- `Type`: What type of text it is
- `Prompt`: What is a relevant rewrite prompt or edit instruction for text
- `Subject`: What is the subject of the prompt, a few words describing the rewrite task, the main idea of it

I will provide some examples followed by the input text for which you will complete the requested information as provided.

Begin!

{fewshot_prompt}

Input Text:
""""""
{input_text}
""""""

### Info
'''

    return prompt


def parse_result(result):
    result = result.replace("### Info", "").strip()
    result = [x for x in result.split("\n") if x.strip() != ""]
    text_type = result[0].split(": ")[1]
    prompt = result[1].split(": ")[1]
    subject = result[2].split(": ")[1]

    return {
        "type": text_type,
        "prompt": prompt,
        "subject": subject
    }

def get_results(input_text, num_fewshot=5, api_params={}):
    content = copy.deepcopy(gpt_conditioned_prompts_base)
    
    np.random.shuffle(content)
    content = content[:num_fewshot]

    prompt = build_prompt(input_text, content)

    result = call_gpt_api(prompt, **api_params)

    try:
        result = parse_result(result)
    except:
        print("INVALID OUTPUT:\n", result, "\n")
        result = {
            "type": "",
            "prompt": "",
            "subject": ""
        }

    return result


out_path = Path(f"/kaggle/input/gpt_conditioned_prompts")
out_path.mkdir(parents=True, exist_ok=True)
dataset_path = out_path / "dataset.csv"

# dataset = load_datasets(min_len=300, max_len=2500, max_size=200)
# dataset.to_csv(dataset_path, index=False)
# print(dataset)
# import pdb; pdb.set_trace()

dataset = pd.read_csv(dataset_path)

api_params = {"system_prompt": system_prompt, "model": "gpt-3.5-turbo", "temperature": 0.5}
results = []

for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    result = get_results(row["original_text"], num_fewshot=5, api_params=api_params)
    results.append(result)

keys = results[0].keys()
results = {key: [dct[key] for dct in results] for key in keys}

proc_dataset = pd.concat([dataset, pd.DataFrame(results)], axis=1)


time_id = time.strftime("%Y%m%d_%H%M%S")
proc_dataset_path = out_path / "proc_dataset" / f"{time_id}.csv"
proc_dataset_path.parent.mkdir(parents=True, exist_ok=True)

proc_dataset.to_csv(proc_dataset_path, index=False)
