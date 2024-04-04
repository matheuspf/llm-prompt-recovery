from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


device = "cuda:0"


def get_embds_score(t5, pred, gt):
    pred_embds = t5.encode(pred, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    gt_embds = t5.encode(gt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)

    res = abs((cosine_similarity(gt_embds, pred_embds)) ** 3)

    return res[0][0]


def filter_df(df):
    df = df.fillna("")

    if "subject" in df.columns:
        df = df[["original_text", "rewrite_prompt", "rewritten_text", "subject"]].reset_index(drop=True)
    
    else:
        df = df[["original_text", "rewrite_prompt", "rewritten_text"]].reset_index(drop=True)
    
    df["original_text"] = df["original_text"].apply(lambda x: str(x).strip())
    df["rewritten_text"] = df["rewritten_text"].apply(lambda x: str(x).strip())
    df["rewrite_prompt"] = df["rewrite_prompt"].apply(lambda x: str(x).strip())

    if "subject" in df.columns:
        df["subject"] = df["subject"].apply(lambda x: str(x).strip())
        df = df[df["subject"].apply(lambda x: len(x) >= 5 and len(x) <= 200)].reset_index(
                drop=True
            )

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(
        drop=True
    )
    df = df[df["rewritten_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(
        drop=True
    )
    df = df[df["rewrite_prompt"].apply(lambda x: len(x) >= 5 and len(x) <= 500)].reset_index(
        drop=True
    )


    return df



def get_dataset_pub(data_path="/kaggle/input/df_with_emb.parquet"):
    df = pd.read_parquet(data_path).fillna("")
    df = df[["original_text", "rewrite_prompt", "rewritten_text"]].reset_index(drop=True)
    df = filter_df(df)
    return df


def get_dataset_gpt():
    data_list = [
        # "/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated.csv",
        # "/kaggle/input/pedro-data/data_subject.csv",
        # "/kaggle/input/pedro-data/data_subject_2.csv",
        # "/kaggle/input/pedro-data/data_subject_3.csv",
        
        "/home/mpf/code/kaggle/llm-prompt/selected_df_optim.csv"
    ]
    df = pd.concat([pd.read_csv(data) for data in data_list], ignore_index=True)
    df = filter_df(df)

    return df


def get_embds_path(t5, text_list, path, batch_size=8):
    path = Path(path)

    # if path.exists():
    if 0:
        return np.load(path, allow_pickle=True)
    
    
    # text_list = ["".join([t for t in text if t.isalpha() or t in (" ",)]) for text in text_list]
    # print(text_list[:10])

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=8)
    np.save(path, embds, allow_pickle=True)

    return embds


def calc_score(t5, prompt, embds):
    prompt_embds = t5.encode(prompt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    res = ((cosine_similarity(embds, prompt_embds)) ** 3).mean()
    return res


def get_dataset_pedro():
    # prompts = json.load(open("/home/mpf/code/kaggle/pedro-llm-prompt/data/prompts_selected.json"))
    prompts = json.load(open("/home/mpf/code/kaggle/pedro-llm-prompt/data/prompts_selected_new.json"))
    df = pd.DataFrame({"rewrite_prompt": prompts})
    return df


def get_dataset_pedro_lowercase():
    prompts = json.load(open("/home/mpf/code/kaggle/pedro-llm-prompt/data/prompts_selected_processed.json"))
    df = pd.DataFrame({"rewrite_prompt": prompts})
    return df



def get_top_words(text_list):
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
    # sorted_bow = sorted_bow[:1000]

    all_words = [tup[0] for tup in sorted_bow]
    # all_words = [w for w in all_words if w not in ("portrayal", "conveying", "convey", "compelling", "compel", "expressing", "improving", "retell", "reword", "engaging", "storytelling", "person", "to")]
    all_words += [",", ".", ":"]

    return all_words



def get_text(words_list):
    text = " ".join(words_list)
    # return text

    text = text[0].upper() + text[1:]
    
    if len(words_list) >= 3:
        text = text + "."
        # text = text + " to ."
    
    return text



def optimize_prompt(t5, embds, top_words, beam_width=50, num_steps=15, batch_size=256):
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)
    
    all_beams = [([], 0)]
    
    best_step_result = []

    for step in range(num_steps):
        new_beams = []
        for sel_words, score in all_beams:
            all_text = [get_text(sel_words + [word]) for word in top_words]
            text_embds = t5.encode(all_text, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
            scores = (cosine_similarity(embds, text_embds) ** 3).mean(axis=0)
            for i, new_score in enumerate(scores):
                new_beams.append((sel_words + [top_words[i]], new_score))

        all_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        best = all_beams[0]
        print(step, best[1], get_text(best[0]))

        result = {"score": float(best[1]), "text": get_text(best[0])}
        best_step_result.append(result)

    return best_step_result


df = get_dataset_pedro()
text_list = df["rewrite_prompt"].tolist()


t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

# text_list = ["".join([t for t in text if t.isalpha() or t in (" ",)]) for text in text_list]
embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=512)
top_words = get_top_words(text_list)

best_step_result = optimize_prompt(t5, embds, top_words, beam_width=100, num_steps=50, batch_size=512)
best = best_step_result[-1]

print(best)
print(calc_score(t5, best["text"], embds))

# with open("./src/optimization/mean_prompt_sel_words.json", "w") as f:
with open("./src/optimization/mean_prompt_all_words.json", "w") as f:
    json.dump(best_step_result, f, indent=4)

# with open("./src/optimization/mean_prompt_all_words_alpha.json", "w") as f:
# with open("./src/optimization/mean_prompt_sel_words_alpha.json", "w") as f:
#     json.dump(best_step_result, f, indent=4)


