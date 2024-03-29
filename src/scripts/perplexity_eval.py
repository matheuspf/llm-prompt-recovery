import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.utils.exllama_utils import ExLLamaModel, Perplexity, get_gemma_prompt

from ..src.utils.data import *

np.random.seed(42)


def build_prompt(row, prompt):
    prompt_ = f"""<start_of_turn>user
{prompt}
{row["original_text"]}<end_of_turn>
<start_of_turn>model
{row["rewritten_text"]}<end_of_turn>"""
    return prompt_


def score(df):
    scs = lambda row: abs((cosine_similarity(row["gt_embds"], row["pred_embds"])) ** 3)
    df["score"] = df.apply(scs, axis=1)

    return np.mean(df["score"])[0][0]


def eval_score_t5(t5, pred, gt):
    pred_embds = t5.encode(pred, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    gt_embds = t5.encode(gt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)

    res = abs((cosine_similarity(gt_embds, pred_embds)) ** 3)

    return res[0][0]


def infer(model, prompt):
    ids = model.tokenizer.encode(
        prompt, encode_special_tokens=True, return_offsets=False, add_bos=True
    )
    # return ids, model.model.forward(ids, cache=None).cpu().float()

    model_start_token = model.tokenizer.encode("<start_of_turn>")[0]
    bos_num_tokens = 3  # <bos><start_of_turn>user
    answer_start = (
        (ids[0] == model_start_token)[bos_num_tokens:].long().argmax().item() + bos_num_tokens + 3
    )

    model.cache.current_seq_len = 0
    model.model.forward(ids[:, :answer_start], model.cache, preprocess_only=True)
    logits = model.model.forward(ids[:, answer_start:], model.cache).float().cpu()

    return ids, logits


data_org = load_data()
embds_org = dict(np.load("/kaggle/input/embeddings.npz", allow_pickle=True))

keys = ["gemma_gen"]

data = pd.concat([data_org[key] for key in keys], ignore_index=True)
embds = np.concatenate([embds_org[key] for key in keys], axis=0)


sel_ids = list(range(100))
data = data.iloc[sel_ids].reset_index(drop=True)
embds = embds[sel_ids]


prompts = data["rewrite_prompt"].tolist()
unique_prompts = sorted(list(set(prompts)))
unique_ids = []

for prompt in unique_prompts:
    idx = prompts.index(prompt)
    unique_ids.append(idx)

data = data.iloc[unique_ids].reset_index(drop=True)
prompts = [prompts[idx] for idx in unique_ids]
embds = embds[unique_ids]
embds = embds / np.linalg.norm(embds, axis=1)[:, None]


model = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2")

perp = Perplexity()

t5_name = "sentence-transformers/sentence-t5-base"
t5 = SentenceTransformer(t5_name, device="cuda")


model_start_token = model.tokenizer.encode("<start_of_turn>")[0]
bos_num_tokens = 3  # <bos><start_of_turn>user
top_k = 15

preds = []
scores = []


prompts = [
    "Please improve the following text using the writing style of {xxx}, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style."
]
# prompts = ["Rewrite, Translate, Improve, Rephrase or Reframe the text"]
embds = t5.encode(prompts, normalize_embeddings=True, show_progress_bar=False)


# prompts = data_org["gpt4_ex1"]["rewrite_prompt"].tolist()
# embds = embds_org["gpt4_ex1"]
# embds = embds / np.linalg.norm(embds, axis=1)[:, None]


def get_mean_prompts(idx):
    es = embds[idx]
    ps = [prompts[ii] for ii in idx]

    # sim_mat = np.dot(es, es.T) ** 3
    # embds_metric = np.mean(sim_mat, axis=1)
    # embds_top_idx = np.argsort(-embds_metric)

    embds_mean = es.mean(axis=0)
    embds_metric = np.linalg.norm(es - embds_mean, axis=1)
    embds_top_idx = np.argsort(embds_metric)

    ps = [ps[idx] for idx in embds_top_idx]

    return ps


for idx, row in tqdm(list(data.iterrows())):
    perps = []

    for prompt in tqdm(prompts):
        prompt = build_prompt(row, prompt)
        ids, logits = infer(model, prompt)

        answer_start = (
            (ids[0] == model_start_token)[bos_num_tokens:].long().argmax().item()
            + bos_num_tokens
            + 3
        )
        # output_answer = logits[:, answer_start:-1]
        output_answer = logits[:, :-1]
        labels_answer = ids[:, answer_start:-1]

        p = perp(output_answer, labels_answer).item()
        perps.append(p)

    perps = np.array(perps)
    best_idx = np.argsort(perps)
    # best_idx = np.array([ii for ii in best_idx if ii != idx])
    best_idx = best_idx[:top_k]

    # best_prompts = [prompts[idx] for idx in best_idx]
    best_prompts = get_mean_prompts(best_idx)

    pred = best_prompts[0]
    preds.append(pred)

    xxx = np.random.choice([x for x in row["rewrite_prompt"].split(" ") if len(x) >= 5])
    pred = pred.replace("{xxx}", xxx)

    # pred = f"{' '.join(row['rewrite_prompt'].split(' ')[:3])} {' '.join(pred.split(' ')[3:])}"

    score = eval_score_t5(t5, pred, row["rewrite_prompt"])

    scores.append(score)

    print(
        "GT:   ",
        row["rewrite_prompt"],
        "\n",
        "-" * 50,
        "\nPred: ",
        pred,
        "\n",
        "Score: ",
        score,
        "\n",
        "=" * 50,
        "\n\n",
    )


print("Mean score: ", np.mean(scores))
