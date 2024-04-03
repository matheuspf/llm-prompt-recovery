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


from optuna import logging
from optuna.pruners import BasePruner


class DuplicateIterationPruner(BasePruner):
    """
    DuplicatePruner

    Pruner to detect duplicate trials based on the parameters.

    This pruner is used to identify and prune trials that have the same set of parameters
    as a previously completed trial.
    """

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

        for completed_trial in completed_trials:
            if completed_trial.params == trial.params:
                return True

        return False


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


t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

df = get_dataset_gpt()
embds_df = get_embds_path(t5, df["rewrite_prompt"].tolist(), "./llm_prompt_embds_gpt.npy")

bow = {}
num_words = 0
for text in df["rewrite_prompt"]:
    word_list = text.lower().split()
    num_words += len(word_list)
    for word in word_list:
        word = "".join([x for x in word if x.isalpha()]).lower()
        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1

for text in bow:
    bow[text] /= num_words

bow_tups = sorted([(k, v) for k, v in bow.items()], key=lambda x: x[1], reverse=True)
bow_tups = bow_tups[:20]
bow_acc_freq = [bow_tups[0][1]]

for i in range(1, len(bow_tups)):
    bow_acc_freq.append(bow_acc_freq[-1] + bow_tups[i][1])

max_word_size = 7

def objective(trial: optuna.Trial):
    word_ids = [trial.suggest_int(f"word_id_{i}", 0, len(bow_tups) - 1) for i in range(max_word_size)]
    text = " ".join([bow_tups[word_id][0] for word_id in word_ids])
    text = text[0].upper() + text[1:] + "."

    embds = t5.encode(text, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    metric = ((embds_df["embds"] @ embds.T) ** 3).mean()

    trial.set_user_attr("text", text)

    return metric



# study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
study = optuna.create_study(direction="maximize", pruner=DuplicateIterationPruner())
study.optimize(objective, n_trials=1000)

trial = study.best_trial

print("Results: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
print("Text : {}".format(trial.user_attrs["text"]))

