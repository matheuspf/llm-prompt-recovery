from itertools import permutations
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from ..optimization.data import *



def get_lower_case(text_list):
    text_list = ["".join([t for t in text if t.isalnum() or t in (" ",)]) for text in text_list]
    return text_list


def get_beam_search_prompts():
    base_path = Path("/home/mpf/code/kaggle/llm-prompt/src/optimization")
    files = sorted(list(base_path.glob("*.json")))
    dcts = sum([json.load(open(f)) for f in files], [])
    text_list = [dct["text"] for dct in dcts]
    return text_list
    
    


def get_df_train(column_name="text"):
    sel_prompts = get_dataset_pedro_lowercase()["rewrite_prompt"].tolist()
    pub_prompts = get_dataset_pub()["rewrite_prompt"].tolist()
    gpt_prompts = get_dataset_gpt()["rewrite_prompt"].tolist()
    beam_search_prompts = get_beam_search_prompts()
    all_prompts = list(set(sel_prompts + pub_prompts + gpt_prompts + beam_search_prompts))
    all_prompts = get_lower_case(all_prompts)

    print(len(all_prompts))
    print("\n\n".join(all_prompts[:5]))

    df_prompts = pd.DataFrame({"text": all_prompts})

    return df_prompts
    


df_train = get_df_train()
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

val_sz = 500
df_val = df_train.iloc[-val_sz:].copy().reset_index(drop=True)
df_train = df_train.iloc[:-val_sz].copy().reset_index(drop=True)


train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

dataset_dict.save_to_disk("/kaggle/input/vec2text-prompt-corpus")
