import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

tqdm.pandas()


def score(df):
    scs = lambda row: abs((cosine_similarity(row["gt_embds"], row["pred_embds"])) ** 3)
    df["score"] = df.apply(scs, axis=1)

    return np.mean(df["score"])[0][0]


model_name = "sentence-transformers/sentence-t5-base"

model = SentenceTransformer(model_name)


df = pd.read_csv("/kaggle/input/gemma-rewrite-nbroad/nbroad-v1.csv")

df["pred"] = ["Improve the text to this."] * len(df)

encode_fn = lambda x: model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(
    1, -1
)

df["pred_embds"] = df["rewrite_prompt"].progress_apply(encode_fn)
df["gt_embds"] = df["pred"].progress_apply(encode_fn)


print(f"CV Score: {score(df)}")


# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/sentence-t5-base')
# embeddings = model.encode(sentences)
# print(embeddings)
