import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.exllama_utils import ExLLamaModel


def load_datasets(base_path="/kaggle/input/gpt_conditioned_prompts/proc_dataset"):
    csv_list = sorted(list(Path(base_path).glob("*.csv")))
    # df = pd.read_csv(csv_list[0])
    df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(
        drop=True
    )

    return df


def load_datasets_free_prompt(base_path="/kaggle/input/gpt_conditioned_prompts/proc_dataset"):
    iters = 1
    base_df = load_datasets(base_path=base_path)
    new_dfs = []

    json_files = sorted(list(Path("/kaggle/input/gpt_free_prompts").glob("*.json")))
    json_files = [f for f in json_files if f.name != "dataset-metadata.json"]
    json_data = sum([json.load(open(json_file)) for json_file in json_files], [])

    for ii in range(iters):
        np.random.shuffle(json_data)
        new_df = base_df.copy()
        new_df["prompt"] = [dct["prompt"] for dct in json_data[: len(new_df)]]
        new_df["subject"] = [dct["subject"] for dct in json_data[: len(new_df)]]
        new_dfs.append(new_df)

    df = pd.concat([base_df] + new_dfs, ignore_index=True)

    return df


def proc_output(output):
    output = output.strip()

    if output.lower().startswith("sure"):
        output = "\n".join(output.split("\n")[1:]).strip()

    return output


model = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2")


# USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
USER_CHAT_TEMPLATE = """<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""

rewrite_data = []


# data = load_datasets()
data = load_datasets_free_prompt()
# data = data.head(10)
data["rewrite_prompt"] = data["prompt"]
data = data.drop("prompt", axis=1)

rewritten_text_list = []


for idx, row in tqdm(data.iterrows(), total=len(data)):
    original_text = row["original_text"]
    rewrite_prompt = row["rewrite_prompt"]
    prompt = f"{rewrite_prompt}\n\n{original_text}"
    user_prompt = USER_CHAT_TEMPLATE.format(prompt=prompt)

    # rewritten_text = model.generate(user_prompt, device=device, output_len=512, top_k=1, temperature=0.1)
    rewritten_text = model.generate(user_prompt, max_tokens=512)
    rewritten_text = proc_output(rewritten_text)

    print(user_prompt, "-" * 100, "\n")
    print(rewritten_text, "\n", len(rewritten_text), "=" * 100, "\n\n")

    rewritten_text_list.append(rewritten_text)


data["rewritten_text"] = rewritten_text_list

output_path = Path("/kaggle/input/gemma_rewritten_text_exllama")
output_path.mkdir(parents=True, exist_ok=True)

data.to_csv(output_path / f"proc_dataset_updated.csv", index=False)
