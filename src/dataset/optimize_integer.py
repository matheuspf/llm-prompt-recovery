import pulp as pl
import numpy as np
import numpy as np
import math
import optuna
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



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
        df["subject"] = ["------------------" for _ in range(df.shape[0])]
    
    df["original_text"] = df["original_text"].apply(lambda x: str(x).strip())
    df["rewritten_text"] = df["rewritten_text"].apply(lambda x: str(x).strip())
    df["rewrite_prompt"] = df["rewrite_prompt"].apply(lambda x: str(x).strip())
    df["subject"] = df["subject"].apply(lambda x: str(x).strip())

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(
        drop=True
    )
    df = df[df["rewritten_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(
        drop=True
    )
    df = df[df["rewrite_prompt"].apply(lambda x: len(x) >= 5 and len(x) <= 500)].reset_index(
        drop=True
    )
    df = df[df["subject"].apply(lambda x: len(x) >= 5 and len(x) <= 200)].reset_index(
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
        "/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_updated.csv",
        # "/kaggle/input/pedro-data/data_subject.csv",
        # "/kaggle/input/pedro-data/data_subject_2.csv",
        # "/kaggle/input/pedro-data/data_subject_3.csv",
    ]
    df = pd.concat([pd.read_csv(data) for data in data_list], ignore_index=True)
    df = filter_df(df)

    return df


def get_prompt_list():
    df = pd.read_csv("/kaggle/input/llm-prompt-recovery-mean-prompts/mean_prompts.csv")
    return df


def get_embds_path(t5, text_list, path, load=True, cluster=True):
    path = Path(path)

    if load and path.exists():
    # if 0:
        return np.load(path, allow_pickle=True)

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True)
    
    np.save(path, embds)

    return embds


def calc_score(t5, prompt, embds):
    prompt_embds = t5.encode(prompt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    res = ((cosine_similarity(embds, prompt_embds)) ** 3).mean()
    return res



# df_pub = get_dataset_pub()
# df_gpt = get_dataset_gpt()
# df = pd.concat([df_pub, df_gpt], ignore_index=True)

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

df = get_dataset_gpt()
embds_df = get_embds_path(t5, df["rewrite_prompt"].tolist(), "./llm_prompt_embds_gpt.npy")

prompts = get_prompt_list()
embds_prompts = get_embds_path(t5, prompts["rewrite_prompt"], "./llm_prompt_embds.npy", cluster=False)
lb_score = prompts["lb_score"].values.astype(np.float32)
lb_score = lb_score ** (1/3)

num_constraints = 5
embds_prompts = embds_prompts[:num_constraints]
lb_score = lb_score[:num_constraints]

A = embds_prompts
B = embds_df
C = lb_score

M, N = A.shape[0], B.shape[0]
delta = 1.0

problem = pl.LpProblem("MaxSubsetWithinRange", pl.LpMaximize)

x = pl.LpVariable.dicts("x", range(N), cat='Binary')

problem += pl.lpSum([x[i] for i in range(N)]), "MaximizeSubsetSize"

for j in range(M):
    C_lower = C[j] - delta
    C_upper = C[j] + delta
    
    problem += pl.lpSum([B[i, :] @ A[j, :].T * x[i] for i in range(N)]) >= C_lower, f"LowerBound_{j}"
    problem += pl.lpSum([B[i, :] @ A[j, :].T * x[i] for i in range(N)]) <= C_upper, f"UpperBound_{j}"

problem.solve(pl.PULP_CBC_CMD(msg=1))

selected_indices = [i for i in range(N) if x[i].varValue > 0.5]
print(f"Selected subset size: {len(selected_indices)}")
print(selected_indices)