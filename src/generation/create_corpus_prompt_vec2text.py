from itertools import permutations
from joblib import Parallel, delayed
from multiprocessing import Pool
from tqdm import tqdm
import copy
import numpy as np
import pickle
from pathlib import Path
import json

import pandas as pd
from datasets import Dataset, DatasetDict
from ..optimization.data import get_dataset_gpt, get_dataset_pedro_lowercase, get_dataset_pub
from ..optimization.mean_prompt_tokens import get_top_words_base, get_synonyms, clean_text



def get_lower_case(text_list):
    text_list = ["".join([t for t in text if t.isalnum() or t in (" ",)]) for text in text_list]
    return text_list


def get_beam_search_prompts():
    base_path = Path("/home/mpf/code/kaggle/llm-prompt/src/optimization")
    files = sorted(list(base_path.glob("*.json")))
    dcts = sum([json.load(open(f)) for f in files], [])
    text_list = [dct["text"] for dct in dcts]
    return text_list
    
    

# def get_df_train(column_name="text"):
#     sel_prompts = get_dataset_pedro_lowercase()["rewrite_prompt"].tolist()
#     pub_prompts = get_dataset_pub()["rewrite_prompt"].tolist()
#     gpt_prompts = get_dataset_gpt()["rewrite_prompt"].tolist()
#     beam_search_prompts = get_beam_search_prompts()
#     all_prompts = list(set(sel_prompts + pub_prompts + gpt_prompts + beam_search_prompts))
#     all_prompts = get_lower_case(all_prompts)

#     print(len(all_prompts))
#     print("\n\n".join(all_prompts[:5]))

#     df_prompts = pd.DataFrame({"text": all_prompts})

#     return df_prompts





def get_top_words():
    words_base = get_top_words_base(get_dataset_pedro_lowercase()["rewrite_prompt"].tolist())
    words_pub = get_top_words_base(get_dataset_pub()["rewrite_prompt"].tolist())
    words_gpt = get_top_words_base(get_dataset_gpt()["rewrite_prompt"].tolist())
    common_words = open("/kaggle/input/common_words.txt", "r").read().split("\n")
    synonyms = get_synonyms()

    all_words = list(set(words_base + words_pub + words_gpt + common_words + synonyms))
    all_words = [clean_text(word) for word in all_words]
    all_words = list(set(all_words))

    return all_words


def get_words_list_shuffle(org_words_list, top_words, num_augs=3):
    all_words_list = [org_words_list]

    for aug_idx in range(num_augs):
        words_list = copy.deepcopy(org_words_list)
        np.random.shuffle(words_list)

        num_to_change = np.random.randint(1, len(words_list) // 2)
        change_idx = np.random.choice(len(words_list), num_to_change, replace=False)
        change_words = np.random.choice(top_words, num_to_change)
        
        for idx, word in zip(change_idx, change_words):
            words_list[idx] = word

        all_words_list.append(words_list)
    
    return all_words_list
    

def get_text(words_list):
    return " ".join(words_list)


def get_df_train(column_name="text"):
    all_beams = pickle.load(open("/home/mpf/code/kaggle/llm-prompt/all_beams.pkl", "rb"))
    top_words = get_top_words()

    all_words_list = [words_list for words_list, _, _ in all_beams]
    all_words_list = [words_list for words_list in all_words_list if len(words_list) >= 45 and len(words_list) <= 55]

    # np.random.shuffle(all_words_list)
    # all_words_list = all_words_list[:1000]

    all_words_list = Parallel(n_jobs=16)(delayed(get_words_list_shuffle)(words_list, top_words, num_augs=2) for words_list in tqdm(all_words_list))
    all_text_list = [get_text(words_list) for words_list_list in tqdm(all_words_list) for words_list in words_list_list]
    all_text_list = list(set(all_text_list))

    df = pd.DataFrame({column_name: all_text_list})

    return df
    


df_train = get_df_train()
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

val_sz = 1000
df_val = df_train.iloc[-val_sz:].copy().reset_index(drop=True)
df_train = df_train.iloc[:-val_sz].copy().reset_index(drop=True)


train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

dataset_dict.save_to_disk("/kaggle/input/vec2text-prompt-corpus")
