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

    return json_data


def get_prompt(org_prompt, num_prompts=3):
    system_prompt = "You will rewrite LLM prompts while keeping the main idea of it"
    prompt = \
f'''Given a LLM prompt for a text rewriting task, you will rewrite the prompt while keeping the original idea intact.

Your input will be a prom

- prompt: The original prompt that you will rewrite while keeping the original idea, the subject of the rewrite intact

Your output will be:

- rewritten_prompts: A list of size {num_prompts} containing the rewritten prompts, each different from each other and the original prompt

Pay attention to the prompt content and the specific wording used in the original prompt. Try to diversify the rewritten prompts as much as possible while keeping the basic task the same.

Begin!

prompt: {org_prompt}
'''
    function_dct = {
        "name": "rewrite_prompts",
        "description": "Rewrite a LLM prompt while keeping the original idea intact",
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


def rewrite_prompts(unique_prompts):
    rewritten_prompts = []

    for org_prompt in tqdm(unique_prompts):
        system_prompt, prompt, function_dct = get_prompt(org_prompt, num_prompts=4)
        response = call_gpt_api(prompt, system_prompt=system_prompt, function_dct=function_dct)
        
        if "rewritten_prompts" not in response:
            print("Error:\n\n", prompt)
            continue

        rewritten_prompts.append(
            {
                "original_prompt": org_prompt,
                "rewritten_prompts": response["rewritten_prompts"]
            }
        )

        print("Original prompt: ", org_prompt)
        print("New prompts : ", response["rewritten_prompts"])
        print("-"*100)
    
    return rewritten_prompts



# data = load_datasets()
# unique_prompts = data["prompt"].unique()

data = load_datasets_free_prompt()
unique_prompts = list(set([d["prompt"] for d in data]))

# Paralelize the call to rewrite_prompts

from joblib import Parallel, delayed

n_jobs = 4

rewritten_prompts = Parallel(n_jobs=n_jobs, backend="threading")(
    delayed(rewrite_prompts)(unique_prompts[ii::n_jobs]) for ii in range(n_jobs)
)


print(len(unique_prompts), len(rewritten_prompts))

output_path = Path("/kaggle/input/gpt_prompts_rewrite")
output_path.mkdir(exist_ok=True, parents=True)

# with open(output_path / "conditioned_prompts.json", "w") as f:
with open(output_path / "free_prompts.json", "w") as f:
    json.dump(rewritten_prompts, f, indent=4)
