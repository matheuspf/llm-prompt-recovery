import torch
import copy
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

from nltk.corpus import wordnet as wn


USE_ALPHA = True
BATCH_SIZE = 256
NUM_PROCESSES = 2
LOAD_STATE = True


def clean_text(text):
    return "".join([t for t in text if t.isalpha() or t in (" ",)]).lower()




def get_synonyms():
    words = [
        "reword", "modify", "rewrite", "convey", "exude", "rephrase", "text", "better", "lucrarea", 
        "tone", "style", "discours", "involving", "creatively", "detail", "write", "this", "emulate",
        "casually", "sender", "recompose", "text", "moved", "genre", "evokes", "recreate", "reword",
        "involved", "help", "message", "essay", "creative", "vibe", "formal", "talk", "writer",
        "alter", "resemble", "poetically", "presenting", "story", "involves", "memo", "recommend",
        "it", "this", "reframe", "especially", "occur", "improve", "currently", "form", "standpoint",
        "line", "changing", "contents", "entity", "phrasing", "pattern"
    ]

    # Function to find synonyms using WordNet
    def find_synonyms(word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    # Get synonyms for each word in the list
    synonyms_dict = {word: find_synonyms(word) for word in words}
    words_list = list(synonyms_dict.keys()) + sum(list(synonyms_dict.values()), [])

    return words_list


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


def get_top_words_vocab(t5, embds, frac=0.2):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
    vocab_keys = list(tokenizer.get_vocab().keys())
    vocab_keys = [k for k in vocab_keys if k not in tokenizer.all_special_tokens and k not in ("", "▁")]
    vocab_keys = [k.replace("▁", "") for k in vocab_keys]

    all_words = [clean_text(word) for word in vocab_keys]
    all_words = [w for w in all_words if len(w) > 0]
    all_words = list(set(all_words))

    all_text = [get_text([word]) for word in all_words]
    text_embds = t5.encode(all_text, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    scores = (cosine_similarity(embds, text_embds) ** 3).mean(axis=0)
    top_words_idx = scores.argsort()[::-1]
    
    num_words = int(frac * len(all_words))
    all_words = [all_words[i] for i in top_words_idx[:num_words]]

    return all_words


def get_top_words(text_list, t5, embds, frac=0.2):
    words_base = get_top_words_base(text_list)
    # vocab_words = get_top_words_vocab(t5, embds, frac=frac)
    words_pub = get_top_words_base(get_dataset_pub()["rewrite_prompt"].tolist())
    words_gpt = get_top_words_base(get_dataset_gpt()["rewrite_prompt"].tolist())
    common_words = open("/kaggle/input/common_words.txt", "r").read().split("\n")
    synonyms = get_synonyms()

    # all_words = list(set(words_base + words_pub + words_gpt + vocab_words + common_words + synonyms))
    all_words = list(set(words_base + words_pub + words_gpt + common_words + synonyms))
    all_words = [clean_text(word) for word in all_words]
    all_words = list(set(all_words))
    
    # all_words = all_words + [w + " " for w in all_words]

    return all_words


def get_text(words_list):
    text = " ".join(words_list)
    # text = "".join(words_list)
    return text
    
    # text = text.replace(" ,", ",").replace(" .", ".").replace(" :", ":")

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


def test_iters(t5, embds, beam, top_words, batch_size=128, iters=5):
    for itx in tqdm(range(iters)):
        sel_words, _, prev_score = beam
        new_text_list = []
        
        for i in tqdm(range(len(sel_words))):
            for word in top_words:
                text_cp = copy.deepcopy(sel_words)
                text_cp[i] = word
                new_text_list.append(get_text(text_cp))
        
        all_embds = t5.encode(new_text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
        scores = (cosine_similarity(embds, all_embds) ** 3).mean(axis=0)
        score_idx = scores.argsort()[::-1]
        
        best_score = scores[score_idx[0]]
        best_words = new_text_list[score_idx[0]].split(" ")
        
        beam = (best_words, best_score, get_beam_score(best_words, best_score))

        print("\n")
        print(prev_score, best_score)
        print(best_words)
        print("\n\n")
    
    import pdb; pdb.set_trace()
        

def get_beams(params):
    all_beams, top_words, embds, t5, batch_size, proc_idx = params
    new_beams = []

    pbar = tqdm(all_beams) if proc_idx == 0 else all_beams

    for sel_words, _, _ in pbar:
        all_text = [get_text(sel_words + [word]) for word in top_words]

        # all_text = [t for t in all_text if t not in cache]
        # for t in all_text:
        #     cache[t] = 1
        
        if len(all_text) == 0:
            continue

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
    pbar = tqdm(range(num_steps))

    if LOAD_STATE:
        all_beams = pickle.load(open("state.pkl", "rb"))
        best_step_result = json.load(open("./src/optimization/mean_prompt_sel_tokens_updated_alpha.json", "r"))
        pbar = tqdm(range(len(best_step_result), num_steps))
        print(len(best_step_result), len(all_beams))

    for step in pbar:
        if NUM_PROCESSES > 1:
            num_processes = min(NUM_PROCESSES, len(all_beams))
            all_beams_split = [all_beams[i::num_processes] for i in range(num_processes)]
            params = [(all_beams_split[i], top_words, embds, t5, batch_size, i) for i in range(num_processes)]

            with Pool(processes=num_processes) as p:
                new_beams = sum(p.map(get_beams, params), [])
        
        else:
            new_beams = get_beams((all_beams, top_words, embds, t5, batch_size, 0))

        all_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

        # all_scores = np.array([beam[2] for beam in all_beams])
        # mean_score = all_scores.mean()
        # std_score = all_scores.std()
        # filter_scores_idx = all_scores > mean_score - 0.5 * std_score
        # all_beams = [beam for i, beam in enumerate(all_beams) if filter_scores_idx[i]]

        best = all_beams[0]

        print("\n", step, best[1], get_text(best[0]), len(all_beams), "\n")

        result = {"step": int(step), "score": float(best[1]), "text": get_text(best[0])}
        best_step_result.append(result)

        pickle.dump(all_beams, open("state.pkl", "wb"))
        with open("./src/optimization/mean_prompt_sel_tokens_updated_alpha.json", "w") as f:
            json.dump(best_step_result, f, indent=4, ensure_ascii=False)

    exit()
    return best_step_result


def run():
    device = "cuda:0"

    if USE_ALPHA:
        df = get_dataset_pedro_lowercase()
        text_list = df["rewrite_prompt"].tolist()
        text_list = ["".join([t for t in text if t.isalpha() or t in (" ",)]).lower() for text in text_list]

    else:
        df = get_dataset_pedro()
        text_list = df["rewrite_prompt"].tolist()


    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    top_words = get_top_words(text_list, t5, embds)

    print(f"Num examples: {len(text_list)}")
    print(f"Num words: {len(top_words)}")

    # all_beams = pickle.load(open("state.pkl", "rb"))
    # beam = all_beams[0]
    # test_iters(t5, embds, beam, top_words, batch_size=BATCH_SIZE)
    # exit()

    best_step_result = optimize_prompt(t5, embds, top_words, beam_width=400, num_steps=71, batch_size=BATCH_SIZE)
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
