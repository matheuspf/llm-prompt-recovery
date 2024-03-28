import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.utils.exllama_utils import ExLLamaModel, Perplexity, get_gemma_prompt

tqdm.pandas()


def build_prompt(prompt, original_text, rewritten_text):
    prompt_text = f'{prompt}: """"""{original_text}""""""'

    chat = [
        {"role": "user", "content": prompt_text},
        {"role": "model", "content": rewritten_text},
    ]

    prompt = get_gemma_prompt(chat)

    if prompt.endswith("<start_of_turn>model\n"):
        prompt = prompt[: -len("<start_of_turn>model\n")]

    return prompt


def score(df):
    scs = lambda row: abs((cosine_similarity(row["gt_embds"], row["pred_embds"])) ** 3)
    df["score"] = df.apply(scs, axis=1)

    return np.mean(df["score"])[0][0]


def eval_score_t5(df, model, prompt):
    df["pred"] = [prompt] * len(df)

    encode_fn = lambda x: model.encode(
        x, normalize_embeddings=True, show_progress_bar=False, batch_size=128
    ).reshape(1, -1)

    df["pred_embds"] = df["rewrite_prompt"].progress_apply(encode_fn)
    df["gt_embds"] = df["pred"].progress_apply(encode_fn)

    res = score(df)

    return res


def eval_perplexity(df, model, prompt):
    df["pred"] = [prompt] * len(df)
    encode_fn = lambda row: model.perplexity(
        build_prompt(row["pred"], row["original_text"], row["rewritten_text"])
    )
    perplexity = df.progress_apply(encode_fn, axis=1)
    return perplexity


df = pd.read_csv("/kaggle/input/gemma-rewrite-nbroad/nbroad-v1.csv")

df["prompt"] = df.apply(
    lambda row: build_prompt(row["rewrite_prompt"], row["original_text"], row["rewritten_text"]),
    axis=1,
)
df = df[df["prompt"].apply(len) < 3 * 8192]


gemma = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2")

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base")

prompts_list = [
    "Improve the text to this.",
    "Improve the text to this:",
    "Make this text more formal.",
]

for prompt in prompts_list:
    df["pred"] = [prompt] * len(df)
    df["perp"] = eval_perplexity(df, gemma, prompt)

    print(prompt)
    print(eval_score_t5(df, t5, prompt))
    print(df["perp"].mean(), "\n\n")
    print("\n")
