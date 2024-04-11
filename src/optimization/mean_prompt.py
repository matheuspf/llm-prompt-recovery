import torch
import pickle
from torch.multiprocessing import Pool

from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from .data import *
from fire import Fire


def clean_text(text):
    return "".join([t for t in text if t.isalpha() or t in (" ",)]).lower()


BATCH_SIZE = 512
NUM_PROCESSES = 1
LOAD_FROM_STATE = True


def get_top_words(text_list, t5, embds):
    bow = {}

    for i, text in enumerate(text_list):
        words = text.split()

        for word in words:
            word = clean_text(word).strip()

            if not word:
                continue

            if word not in bow:
                bow[word] = 0

            bow[word] += 1

    bow_tup = [(k, v) for k, v in bow.items()]
    sorted_bow = sorted(bow_tup, key=lambda x: x[1], reverse=True)
    sorted_bow = list(sorted_bow)
    all_words = [tup[0] for tup in sorted_bow]

    return all_words


def get_text(words_list):
    text = " ".join(words_list)
    return clean_text(text)


def get_beam_score(words, embd_score, alpha=0.1):
    # return embd_score / (1.0 + alpha * np.log2(1.0 + len(words)))
    return embd_score


def get_beams(params):
    all_beams, top_words, embds, t5, batch_size, proc_idx = params
    new_beams = []

    pbar = tqdm(all_beams) if proc_idx == 0 else all_beams

    for sel_words, _, _ in pbar:
        all_text = [get_text(sel_words + [word]) for word in top_words]
        text_embds = t5.encode(
            all_text, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size
        )
        scores = (cosine_similarity(embds, text_embds) ** 3).mean(axis=0)

        for i, new_score in enumerate(scores):
            new_words = sel_words + [top_words[i]]
            new_score_beam = get_beam_score(new_words, new_score)
            new_beams.append((new_words, new_score, new_score_beam))

    return new_beams


def optimize_prompt(t5, embds, top_words, beam_width=50, num_steps=15, batch_size=256):
    all_beams = [([], 0, 0)]

    best_step_result = []

    start_idx = 0
    if LOAD_FROM_STATE:
        all_beams = pickle.load(open("state.pkl", "rb"))
        start_idx = 50

    for step in tqdm(range(start_idx, num_steps)):
        if NUM_PROCESSES > 1:
            num_processes = min(NUM_PROCESSES, len(all_beams))
            all_beams_split = [all_beams[i::num_processes] for i in range(num_processes)]
            params = [
                (all_beams_split[i], top_words, embds, t5, batch_size, i)
                for i in range(num_processes)
            ]

            with Pool(processes=num_processes) as p:
                new_beams = sum(p.map(get_beams, params), [])

        else:
            new_beams = get_beams((all_beams, top_words, embds, t5, batch_size, 0))

        print(len(new_beams))

        all_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

        # all_scores = np.array([beam[2] for beam in all_beams])
        # mean_score = all_scores.mean()
        # std_score = all_scores.std()
        # filter_scores_idx = all_scores > mean_score - 0.5 * std_score
        # all_beams = [beam for i, beam in enumerate(all_beams) if filter_scores_idx[i]]

        best = all_beams[0]

        print("\n", step, best[1], get_text(best[0]), len(all_beams), "\n")

        result = {"score": float(best[1]), "text": get_text(best[0])}
        best_step_result.append(result)

        pickle.dump(all_beams, open("state.pkl", "wb"))

        with open(f"./src/optimization/mean_prompt_alpha_cluster_{CLUSTER}.json", "w") as f:
            json.dump(best_step_result, f, indent=4)

    return best_step_result


def run():
    device = "cuda:0"

    df = get_dataset_pedro_lowercase()
    text_list = df["rewrite_prompt"].tolist()
    text_list = [clean_text(text) for text in text_list]

    df_pub = get_dataset_pub()
    df_gpt = get_dataset_gpt()

    extra_text_list = (
        text_list + df_pub["rewrite_prompt"].tolist() + df_gpt["rewrite_prompt"].tolist()
    )
    extra_text_list = [clean_text(text) for text in extra_text_list]

    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    target_prompts = pd.read_csv(
        "/home/edu/code/llm_style/data/cluster_models/2.6k_selected_prompts_with_clusters_2_3.csv"
    )
    target_prompts = target_prompts[target_prompts["cluster_3"] == CLUSTER]
    target_prompts = target_prompts["rewrite_prompt"].tolist()
    target_prompts = [clean_text(text) for text in target_prompts]
    embds = t5.encode(
        target_prompts, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE
    )

    top_words = get_top_words(extra_text_list, t5, embds)

    print(f"Num examples: {len(text_list)}")
    print(f"Num words: {len(top_words)}")

    best_step_result = optimize_prompt(
        t5, embds, top_words, beam_width=400, num_steps=100, batch_size=BATCH_SIZE
    )
    best = best_step_result[-1]

    print(best)
    print(calc_score(t5, best["text"], embds))

    with open(f"./src/optimization/mean_prompt_alpha_cluster_{CLUSTER}.json", "w") as f:
        json.dump(best_step_result, f, indent=4)


def main(cluster):
    global CLUSTER
    CLUSTER = cluster
    torch.multiprocessing.set_start_method("spawn")
    run()


if __name__ == "__main__":
    Fire(main)
