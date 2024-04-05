from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .data import *


USE_ALPHA = False


def get_top_words(text_list):
    new_text_list = []
    
    bow = {}

    for i, text in enumerate(text_list):    
        words = text.split()
        
        for word in words:
            word = "".join(filter(str.isalnum, word)).lower().strip()
            
            if not word:
                continue

            if word not in bow:
                bow[word] = 0
            
            bow[word] += 1
            
    bow_tup = [(k, v) for k, v in bow.items()]
    sorted_bow = sorted(bow_tup, key=lambda x: x[1], reverse=True)
    print(len(sorted_bow))
    sorted_bow = list(sorted_bow)

    all_words = [tup[0] for tup in sorted_bow]
    # all_words = [w for w in all_words if w not in ("portrayal", "conveying", "convey", "compelling", "compel", "expressing", "improving", "retell", "reword", "engaging", "storytelling", "person", "to")]

    if not USE_ALPHA:
        all_words += [",", ".", ":"]

    return all_words



def get_text(words_list):
    text = " ".join(words_list)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" :", ":")

    if USE_ALPHA:
        return text

    text = text[0].upper() + text[1:]
    
    if len(words_list) >= 3:
        text = text + "."
        # text = text + " to ."
    
    return text


def get_beam_score(words, embd_score, alpha=0.1):
    # return embd_score / (1.0 + alpha * np.log2(1.0 + len(words)))
    return embd_score


def optimize_prompt(t5, embds, top_words, beam_width=50, num_steps=15, batch_size=256):
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)
    
    all_beams = [([], 0, 0)]
    cache = {}
    
    best_step_result = []

    for step in range(num_steps):
        new_beams = []
        for sel_words, _, _ in all_beams:
            all_text = [get_text(sel_words + [word]) for word in top_words]
            all_text = [t for t in all_text if t not in cache]

            for t in all_text:
                cache[t] = 1

            text_embds = t5.encode(all_text, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
            scores = (cosine_similarity(embds, text_embds) ** 3).mean(axis=0)
            for i, new_score in enumerate(scores):
                new_words = sel_words + [top_words[i]]
                new_score_beam = get_beam_score(new_words, new_score)
                new_beams.append((new_words, new_score, new_score_beam))

        all_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

        all_scores = np.array([beam[2] for beam in all_beams])
        mean_score = all_scores.mean()
        std_score = all_scores.std()
        filter_scores_idx = all_scores > mean_score - 0.5 * std_score

        all_beams = [beam for i, beam in enumerate(all_beams) if filter_scores_idx[i]]
        best = all_beams[0]

        print(step, best[1], get_text(best[0]), len(all_beams))

        result = {"score": float(best[1]), "text": get_text(best[0])}
        best_step_result.append(result)

    return best_step_result


def run():
    device = "cuda:0"

    if USE_ALPHA:
        df = get_dataset_pedro_lowercase()
        text_list = df["rewrite_prompt"].tolist()
        text_list = ["".join([t for t in text if t.isalpha() or t in (" ",)]) for text in text_list]

    else:
        df = get_dataset_pedro()
        text_list = df["rewrite_prompt"].tolist()


    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    top_words = get_top_words(text_list)


    print(f"Num examples: {len(text_list)}")
    print(f"Num words: {len(top_words)}")

    best_step_result = optimize_prompt(t5, embds, top_words, beam_width=100, num_steps=50, batch_size=64)
    best = best_step_result[-1]

    print(best)
    print(calc_score(t5, best["text"], embds))

    if USE_ALPHA:
        with open("./src/optimization/mean_prompt_sel_words_alpha.json", "w") as f:
            json.dump(best_step_result, f, indent=4)

    else:
        with open("./src/optimization/mean_prompt_sel_words_updated.json", "w") as f:
            json.dump(best_step_result, f, indent=4)


if __name__ == "__main__":
    run()

