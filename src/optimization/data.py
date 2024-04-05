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



def get_dataset_pub(data_path="/kaggle/input/df_with_emb.csv"):
    df = pd.read_csv(data_path).fillna("")
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
    # prompts = json.load(open("/home/mpf/code/kaggle/pedro-llm-prompt/data/prompts_selected_processed.json"))
    prompts = json.load(open("/home/mpf/code/kaggle/pedro-llm-prompt/data/prompts_selected_new_processed_005.json"))
    df = pd.DataFrame({"rewrite_prompt": prompts})
    return df


def get_gen_sel_dataset():
    df = pd.read_csv("/kaggle/input/fitted_conditioned_prompts/df_prompts.csv")
    return df

