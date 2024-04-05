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
from itertools import permutations, chain

def get_permutations(x, max_length):
    return list(chain.from_iterable(permutations(x, r) for r in range(1, max_length + 1)))


def get_mean_prompt_example():
    return "Rewrite this text convey manner human evokes text better exude genre plath tone cut include object being about please further wise this individuals could originally convey here."


USE_ALPHA = True
BATCH_SIZE = 512
NUM_PROCESSES = 4



def clean_text(text):
    return "".join([t for t in text if t.isalpha() or t in (" ",)]).lower()


def find_optimal_subject(t5, mean_prompt, df, max_words=3, max_test = 10000):
    best_subjects = []

    mean_prompt = clean_text(mean_prompt) + " {{subject}}"
    scores_list = []
    
    # for idx, row in tqdm(df.iterrows(), total=len(df)):
    for idx, row in df.iterrows():
        prompt = clean_text(row["rewrite_prompt"])

        prompt_words = prompt.split(" ")[::-1]
        prompt_words = get_permutations(prompt_words, max_words)
        prompt_words = sorted(prompt_words, key=lambda x: len(x))[::-1]
        print(prompt)
        print(prompt_words[:5])
        import pdb; pdb.set_trace()
        prompt_words = prompt_words[:max_test]

        prompts = [clean_text(mean_prompt.replace("{{subject}}", " ".join(words))) for words in prompt_words]
        embds = t5.encode(prompts + [prompt], normalize_embeddings=True, show_progress_bar=False, batch_size=512)

        scores = (cosine_similarity(embds[:-1], embds[-1].reshape(1, -1)) ** 3)

        best_words = prompt_words[np.argmax(scores)]
        best_subject = " ".join(best_words)

        print(prompt[:50] + " ...", " | ", best_subject, " | ", np.max(scores))

        scores_list.append(np.max(scores))
        best_subjects.append(best_subject) 

        print(np.mean(scores_list))
    
    df["best_subject"] = best_subjects

    return df 



def get_top_words_base(text_list):
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

    if not USE_ALPHA:
        all_words += [",", ".", ":"]

    return all_words


def get_top_words_vocab(text_list, t5, embds, frac=0.2):
    base_words = get_top_words_base(text_list)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    vocab_keys = list(tokenizer.get_vocab().keys())
    vocab_keys = [k for k in vocab_keys if k not in tokenizer.all_special_tokens and k not in ("", "▁")]
    vocab_keys = [k.replace("▁", "") for k in vocab_keys]

    all_words = base_words + vocab_keys
    all_words = [clean_text(word) for word in all_words]
    all_words = [w for w in all_words if len(w) > 0]

    all_words = list(set(all_words))

    all_text = [get_text([word]) for word in all_words]
    text_embds = t5.encode(all_text, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    scores = (cosine_similarity(embds, text_embds) ** 3).mean(axis=0)
    top_words_idx = scores.argsort()[::-1]
    
    num_words = int(frac * len(all_words))
    all_words = [all_words[i] for i in top_words_idx[:num_words]]

    # all_words = all_words + [w + " " for w in all_words]

    return all_words


def get_text(words_list):
    text = " ".join(words_list)
    return text


def get_beam_score(words, embd_score, alpha=0.1):
    # return embd_score / (1.0 + alpha * np.log2(1.0 + len(words)))
    return embd_score


def get_beams(params):
    all_beams, top_words, embds, t5, batch_size, subjects, proc_idx = params
    new_beams = []

    pbar = tqdm(all_beams) if proc_idx == 0 else all_beams

    for sel_words, _, _ in pbar:
        all_text = [get_text(sel_words + [word]) for word in top_words]

        text_embds = t5.encode(all_text, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
        scores = (cosine_similarity(embds, text_embds) ** 3).mean(axis=0)
        for i, new_score in enumerate(scores):
            new_words = sel_words + [top_words[i]]
            new_score_beam = get_beam_score(new_words, new_score)
            new_beams.append((new_words, new_score, new_score_beam))

    return new_beams


def optimize_prompt(t5, embds, top_words, beam_width=50, num_steps=15, batch_size=256):
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)
    
    all_beams = [([], 0, 0)]
    
    best_step_result = []

    for step in tqdm(range(num_steps)):
        if NUM_PROCESSES > 1:
            num_processes = min(NUM_PROCESSES, len(all_beams))
            all_beams_split = [all_beams[i::num_processes] for i in range(num_processes)]
            params = [(all_beams_split[i], top_words, embds, t5, batch_size, i) for i in range(num_processes)]

            with Pool(processes=num_processes) as p:
                new_beams = p.map(get_beams, params)[0]

        
        else:
            new_beams = get_beams((all_beams, top_words, embds, t5, batch_size, 0))

        all_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

        best = all_beams[0]

        print("\n", step, best[1], get_text(best[0]), len(all_beams), "\n")

        result = {"score": float(best[1]), "text": get_text(best[0])}
        best_step_result.append(result)

        pickle.dump(all_beams, open("state.pkl", "wb"))

        with open("./src/optimization/mean_prompt_sel_tokens_updated_alpha.json", "w") as f:
            json.dump(best_step_result, f, indent=4)

    return best_step_result


def run():
    device = "cuda:0"

    if USE_ALPHA:
        df = get_dataset_pedro_lowercase()
        text_list = df["rewrite_prompt"].tolist()
        text_list = [clean_text(text) for text in text_list]

    else:
        df = get_dataset_pedro()
        text_list = df["rewrite_prompt"].tolist()

    
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    df = find_optimal_subject(t5, get_mean_prompt_example(), df, max_words=3, max_test=10000)

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    top_words = get_top_words_vocab(text_list, t5, embds)

    print(f"Num examples: {len(text_list)}")
    print(f"Num words: {len(top_words)}")

    best_step_result = optimize_prompt(t5, embds, top_words, beam_width=100, num_steps=50, batch_size=BATCH_SIZE)
    best = best_step_result[-1]

    print(best)
    print(calc_score(t5, best["text"], embds))

    if USE_ALPHA:
        with open("./src/optimization/mean_prompt_sel_tokens_alpha.json", "w") as f:
            json.dump(best_step_result, f, indent=4)

    else:
        with open("./src/optimization/mean_prompt_sel_tokens_updated.json", "w") as f:
            json.dump(best_step_result, f, indent=4)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    run()
