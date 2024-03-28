import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(base_dir: str | Path = "/kaggle/input") -> dict[str, pd.DataFrame]:
    return {
        "base_comp": load_base_comp_data(base_dir),
        "nbroad": load_nbroad_data(base_dir),
        "gemma_gen": load_gemma_gen_data(base_dir),
        "gpt4_ex1": load_gpt4_ex1_data(base_dir),
    }


def load_base_comp_data(base_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(base_dir) / "llm-prompt-recovery" / "train.csv")


def load_nbroad_data(base_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(base_dir) / "gemma-rewrite-nbroad" / "nbroad-v2.csv")


def load_gemma_gen_data(base_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(base_dir) / "llm-prompt-recovery-data" / "gemma10000.csv")


def load_gpt4_ex1_data(base_dir: str | Path) -> pd.DataFrame:
    lines = [line.strip() for line in open("/kaggle/input/llm_prompt_gpt4_ex1.txt").readlines()]
    return pd.DataFrame({"rewrite_prompt": lines})


def proc_embeddings(
    data_dict: pd.DataFrame | dict[str, pd.DataFrame],
    output_path: str | Path,
    model_name="sentence-transformers/sentence-t5-base",
):
    if isinstance(data_dict, pd.DataFrame):
        data_dict = {"data": data_dict}

    model = SentenceTransformer(model_name)
    embeddings_dict = {}

    for name, df in data_dict.items():
        prompts = df["rewrite_prompt"].tolist()
        embeddings = model.encode(
            prompts, normalize_embeddings=False, show_progress_bar=True, batch_size=128
        )
        embeddings_dict[name] = embeddings

    output_path = Path(output_path)

    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")

    np.savez(output_path, **embeddings_dict, allow_pickle=True)


def proc_embeddings_path(
    base_dir: str | Path = "/kaggle/input",
    output_path: str | Path = "/kaggle/input/embeddings.npz",
    model_name="sentence-transformers/sentence-t5-base",
):
    data = load_data(base_dir)
    proc_embeddings(data, output_path, model_name)


def get_gen_prompts(base_dir: str | Path = "/kaggle/input") -> pd.DataFrame:
    files = list((Path(base_dir) / "gen_data_test_v2").glob("*.json"))
    data = []

    for f in files:
        with open(f) as f:
            data += json.load(f)["prompts"]

    data = pd.DataFrame(
        {
            "rewrite_prompt": [dct["prompt"] for dct in data],
            "subject": [dct["subject"] for dct in data],
        }
    )
    return data


if __name__ == "__main__":

    # proc_embeddings_path()

    data = load_data()
    embds = dict(np.load("/kaggle/input/embeddings.npz", allow_pickle=True))

    keys = ["gpt4_ex1"]

    data = pd.concat([data[key] for key in keys], ignore_index=True)
    embds = np.concatenate([embds[key] for key in keys], axis=0)

    prompts = data["rewrite_prompt"].tolist()
    unique_prompts = set(prompts)

    unique_ids = []

    for prompt in unique_prompts:
        idx = prompts.index(prompt)
        unique_ids.append(idx)

    prompts = [prompts[idx] for idx in unique_ids]
    embds = embds[unique_ids]

    embds = embds / np.linalg.norm(embds, axis=1, keepdims=True)

    embds_mean = embds.mean(axis=0)
    embds_metric = np.linalg.norm(embds - embds_mean, axis=1)
    embds_top_idx = np.argsort(embds_metric)

    # embds_sim = cosine_similarity(embds)
    # embds_metric = embds_sim.mean(axis=1)
    # embds_top_idx = np.argsort(-embds_metric)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    embds = PCA(n_components=32).fit_transform(embds)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embds)
    embds_metric = kmeans.transform(embds).min(axis=1)
    embds_top_idx = np.argsort(embds_metric)

    for idx in embds_top_idx[:20]:
        # print(f"{embds_metric[idx]:3f} - {prompts[idx]}")
        print(prompts[idx])


# if __name__ == "__main__":
#     fire.Fire()
