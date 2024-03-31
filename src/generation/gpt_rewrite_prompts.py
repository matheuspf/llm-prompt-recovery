import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .gpt_utils import call_gpt_api


def load_datasets(base_path="/kaggle/input/gpt_conditioned_prompts/proc_dataset"):
    csv_list = sorted(list(Path(base_path).glob("*.csv")))
    df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(
        drop=True
    )
    df["id"] = list(range(len(df)))

    return df


def load_datasets_free_prompt(base_path="/kaggle/input/gpt_conditioned_prompts/proc_dataset", num_iters=1):
    base_df = load_datasets(base_path=base_path)
    iters = 1
    new_dfs = []

    json_files = sorted(list(Path("/kaggle/input/gpt_free_prompts").glob("*.json")))
    json_files = [f for f in json_files if f.name != "dataset-metadata.json"]
    json_data = sum([json.load(open(json_file)) for json_file in json_files], [])

    for ii in range(num_iters):
        np.random.shuffle(json_data)
        new_df = base_df.copy()
        new_df["prompt"] = [dct["prompt"] for dct in json_data[: len(new_df)]]
        new_df["subject"] = [dct["subject"] for dct in json_data[: len(new_df)]]
        new_dfs.append(new_df)

    df = pd.concat([base_df] + new_dfs, ignore_index=True)

    return df


def get_prompt(dct, num_prompts=3):
    system_prompt = "You will rewrite LLM prompts while keeping the original subject"
    prompt = \
f'''Given a LLM prompt and the extracted subject from it, you will rewrite the prompt while keeping the original subject intact.

Your input will be:

- prompt: The original prompt that you will rewrite
- subject: The subject extracted from the prompt, a few words describing the core of it

Your output will be:

- rewritten_prompts: A list of size {num_prompts} containing the rewritten prompts, each different from each other and the original prompt

Pay attention to the prompt content and the specific wording used in the original prompt. Try to diversify the rewritten prompts as much as possible while keeping the original subject intact.

Begin!

```
prompt: {dct["rewrite_prompt"]}
subject: {dct["subject"]}
```
'''
    function_dct = {
        "name": "rewrite_prompts",
        "description": "Rewrite a LLM prompt while keeping the original subject intact",
        "parameters": {
            "type": "object",
            "properties": {
                "rewritten_prompts": {
                    "type": "array",
                    "description": f"A list of size {num_prompts} containing the rewritten prompts as described",
                    "items": {
                        "type": "string",
                        "description": "The rewritten prompt, which should be different from the original one"
                    }
                },
            },
            "required": ["rewritten_prompts"],
        },
    }

    return system_prompt, prompt, function_dct


data = load_datasets_free_prompt()
# data = load_datasets()
data["rewrite_prompt"] = data["prompt"]
data = data.drop("prompt", axis=1)

new_rows = []

data = data.head(10)

for idx, row in tqdm(data.iterrows(), total=len(data)):
    system_prompt, prompt, function_dct = get_prompt(row, num_prompts=1)
    response = call_gpt_api(prompt, system_prompt=system_prompt, function_dct=function_dct)
    
    if "rewritten_prompts" not in response:
        print("Error:\n\n", prompt)
        continue

    rewritten_prompts = response["rewritten_prompts"]

    print("Original prompt: ", row["rewrite_prompt"])
    print("New prompts : ", rewritten_prompts)
    print("Subject: ", row["subject"])
    print("-"*100)
    
    for rewritten_prompt in rewritten_prompts:
        new_row = row.copy()
        new_row["rewritten_text"] = rewritten_prompt
        new_rows.append(new_row)

new_data = pd.concat([data] + new_rows, ignore_index=True)

output_path = Path("/kaggle/input/gpt_conditioned_prompts_rewrite")
output_path.mkdir(parents=True, exist_ok=True)
new_data.to_csv(output_path / f"dataset.csv", index=False)
