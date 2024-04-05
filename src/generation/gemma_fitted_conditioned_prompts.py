import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .gpt_utils import call_gpt_api
from .prompts import gpt_conditioned_prompts_base
from .gpt_conditioned_prompts import *
from ..optimization.data import *

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.exllama_utils import ExLLamaModel


LOAD_DATASET = 1

seed = 42
np.random.seed(seed)


def proc_output(output):
    output = output.strip()

    if output.lower().startswith("sure"):
        output = "\n".join(output.split("\n")[1:]).strip()

    return output


def run():
    model = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2")

    USER_CHAT_TEMPLATE = """<start_of_turn>user
    {prompt}<end_of_turn>
    <start_of_turn>model
    """
    
    out_path = Path(f"/kaggle/input/fitted_conditioned_prompts")
    out_path.mkdir(parents=True, exist_ok=True)

    text_dataset_path = out_path / "text_dataset.csv"
    df_prompts_path = out_path / "df_prompts.csv"

    if LOAD_DATASET:
        text_dataset = load_datasets(min_len=300, max_len=2500, max_size=1000)
        text_dataset = text_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
        text_dataset.to_csv(text_dataset_path, index=False)
    
    else:
        text_dataset = pd.read_csv(text_dataset_path)
    
    df_prompts = get_dataset_pedro()

    original_text_list = text_dataset["original_text"].values.tolist()[:len(df_prompts)]
    dataset_id_list = text_dataset["dataset_id"].values.tolist()[:len(df_prompts)]

    df_prompts["original_text"] = original_text_list
    df_prompts["dataset_id"] = dataset_id_list
    
    rewritten_text_list = []

    for idx, row in tqdm(df_prompts.iterrows(), total=len(df_prompts)):
        original_text = row["original_text"]
        rewrite_prompt = row["rewrite_prompt"]

        prompt = f"{rewrite_prompt}\n\n{original_text}"
        user_prompt = USER_CHAT_TEMPLATE.format(prompt=prompt)

        rewritten_text = model.generate(user_prompt, max_tokens=512)
        rewritten_text = proc_output(rewritten_text)

        print(user_prompt, "-" * 100, "\n")
        print(rewritten_text, "\n", len(rewritten_text), "=" * 100, "\n\n")

        rewritten_text_list.append(rewritten_text)

    df_prompts["rewritten_text"] = rewritten_text_list
    df_prompts.to_csv(df_prompts_path, index=False)


if __name__ == "__main__":
    run()
