import sys

sys.path.append("/home/mpf/code/kaggle/gemma_pytorch")

import contextlib
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gemma.config import GemmaConfig, get_config_for_2b, get_config_for_7b
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
from tqdm import tqdm


def load_datasets(base_path="/kaggle/input/gpt_conditioned_prompts/proc_dataset"):
    csv_list = sorted(list(Path(base_path).glob("*.csv")))
    return pd.read_csv(csv_list[0])


def proc_output(output):
    output = output.strip()

    if output.lower().startswith("sure"):
        output = "\n".join(output.split("\n")[1:]).strip()

    return output


# Load the model
VARIANT = "7b-it-quant"
MACHINE_TYPE = "cuda"
weights_dir = "/kaggle/input/gemma/pytorch/7b-it-quant/2"


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

    # Model Config.


model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = "quant" in VARIANT

# Model.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
    model = GemmaForCausalLM(model_config)
    ckpt_path = os.path.join(weights_dir, f"gemma-{VARIANT}.ckpt")
    model.load_weights(ckpt_path)
    model = model.to(device).eval()


# This is the prompt format the model expects
# USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
USER_CHAT_TEMPLATE = """<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""

rewrite_data = []


data = load_datasets()
data = data.head(10)
data["rewrite_prompt"] = data["prompt"]

rewritten_text_list = []


for idx, row in tqdm(data.iterrows(), total=len(data)):
    original_text = row["original_text"]
    rewrite_prompt = row["rewrite_prompt"]
    prompt = f"{rewrite_prompt}\n{original_text}"
    user_prompt = USER_CHAT_TEMPLATE.format(prompt=prompt)

    rewritten_text = model.generate(
        user_prompt, device=device, output_len=512, top_k=1, temperature=0.1
    )
    rewritten_text = proc_output(rewritten_text)

    print(user_prompt, "-" * 100, "\n")
    print(rewritten_text, "\n", len(rewritten_text), "=" * 100, "\n\n")

    rewritten_text_list.append(rewritten_text)


data["rewritten_text"] = rewritten_text_list

output_path = Path("/kaggle/input/gemma_rewritten_text_pytorch")
output_path.mkdir(parents=True, exist_ok=True)

time_id = time.strftime("%Y%m%d_%H%M%S")

data.to_csv(output_path / f"proc_dataset_{time_id}.csv", index=False)
