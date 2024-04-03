import numpy as np
import json
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

    # if load and path.exists():
    if 0:
        return np.load(path, allow_pickle=True).item()

    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True)

    if cluster:
        # Cluster embeddings
        # pca = PCA(n_components=64)
        # embds_pca = pca.fit_transform(embds)
        kmeans = KMeans(n_clusters=len(embds) // 100)
        labels = kmeans.fit_predict(embds)
        centers = kmeans.cluster_centers_

        embds_dict = {
            "embds": embds,
            "labels": labels,
            "centers": centers,
        }
    
    else:
        embds_dict = {
            "embds": embds,
        }
    
    np.save(path, embds_dict)

    return embds_dict


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


# rewritten_prompts = json.load(open("/kaggle/input/gpt_prompts_rewrite/conditioned_prompts.json"))
# rewritten_prompts = sum(rewritten_prompts, [])
# all_prompts = [[dct["original_prompt"]] + dct["rewritten_prompts"] for dct in rewritten_prompts]
# all_prompts = sum(all_prompts, [])
# embds_df = get_embds_path(t5, all_prompts, "./llm_prompt_embds_gpt.npy", cluster=True)



# df = get_dataset_pub()
# embds_df = get_embds_path(t5, df["rewrite_prompt"].tolist(), "./llm_prompt_embds_pub.npy")

prompts = get_prompt_list()
embds_prompts = get_embds_path(t5, prompts["rewrite_prompt"], "./llm_prompt_embds.npy", cluster=False)["embds"]
lb_score = prompts["lb_score"].values.astype(np.float32)


num_constraints = 100
block_size = 10
penalty_frac = 1e2


embds_prompts = embds_prompts[:num_constraints]
lb_score = lb_score[:num_constraints]



def objective(trial: optuna.Trial):
    centers = embds_df["centers"]
    N = math.ceil(len(centers) / block_size)
    
    subset = [trial.suggest_int(f"subset_{i}", 0, block_size**2) for i in range(N)]
    subset = [list(map(int, list(bin(i)[2:].zfill(block_size)))) for i in subset]
    subset = np.array(sum(subset, []), dtype=bool)[:len(centers)]

    sel_centers_idx = np.arange(len(centers))[subset]
    sel_embds_idx = np.isin(embds_df["labels"], sel_centers_idx)
    sel_embds = embds_df["embds"][sel_embds_idx]

    sel_df = df.iloc[sel_embds_idx].copy().reset_index(drop=True)
    scores = (np.matmul(sel_embds, embds_prompts.T) ** 3).mean(axis=0)
    # scores_diff = (((scores - lb_score) * penalty_frac) ** 2).mean()
    scores_diff = (((scores - lb_score) * penalty_frac) ** 2).mean()
    actual_diff = np.abs(scores - lb_score)

    trial.set_user_attr("subset", subset)
    trial.set_user_attr("num_embds", len(sel_embds))
    trial.set_user_attr("scores", scores)
    trial.set_user_attr("scores_diff", scores_diff)
    trial.set_user_attr("actual_diff", actual_diff)
    trial.set_user_attr("lb_score", lb_score)
    trial.set_user_attr("sel_embds", sel_embds)
    trial.set_user_attr("sel_df", sel_df)
    
    # metric = len(sel_embds) - penalty_frac * scores_diff
    # metric = len(sel_embds) - scores_diff
    metric = len(sel_embds) / len(embds_df["embds"]) - scores_diff

    return metric



study = optuna.create_study(direction="maximize")#, pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
study.optimize(objective, n_trials=2000)#, n_jobs=4)

trial = study.best_trial

print("Results: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
print(f"Num embds: {trial.user_attrs['num_embds']}")
print(f"Constraints: {trial.user_attrs['scores']}")
print(f"LB score:", lb_score)
print(f"Constraint diff: {trial.user_attrs['scores_diff']}")


sel_centers_idx = np.arange(len(embds_df["centers"]))[trial.user_attrs["subset"]]
sel_embds_idx = np.isin(embds_df["labels"], sel_centers_idx)
# sel_prompts = np.array(all_prompts)[sel_embds_idx]

optim_output = {
    "subset": trial.user_attrs["subset"],
    "sel_embds": trial.user_attrs["sel_embds"],
    "scores": trial.user_attrs["scores"],
    "scores_diff": trial.user_attrs["scores_diff"],
    "lb_score": trial.user_attrs["lb_score"],
    # "sel_prompts": sel_prompts,
}
np.save("optim_output.npy", optim_output)

# print(len(all_prompts), len(trial.user_attrs["sel_embds"]))

# sel_embds = embds_df["embds"][sel_embds_idx]
# dists = ((sel_embds @ sel_embds.T) ** 3).mean(axis=1)
# dists_idx = np.argsort(dists)
# best_idx = dists_idx[:-10:-1]

# print(dists[best_idx], sel_prompts[best_idx])




sel_df = trial.user_attrs["sel_df"]
sel_df.to_csv("selected_df_optim.csv")

print(len(df), len(sel_df))
