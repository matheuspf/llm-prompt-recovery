import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from .data import *
import vec2text
from textblob import TextBlob


device = "cuda:0"
LR = 1e-3


def get_initial_embds(t5):
    prompt = "improve phrasing text lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea"
    embds = t5.encode(prompt, normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True)
    embds.requires_grad = True
    return embds

def get_data_embds(t5):
    df = get_dataset_pedro_lowercase()
    text_list = df["rewrite_prompt"].tolist()
    data_embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True, batch_size=8)
    return data_embds


def get_data_embds_cluster(t5, cluster_idx):
    df = pd.read_csv("./fitted_conditioned_prompts_with_clusters.csv")
    # df = df[df["cluster_20"] == cluster_idx]

    sel_idx = []
    for idx, row in df.iterrows():
        if len(row["original_text"]) > len(row["rewritten_text"]):
            sel_idx.append(idx)
    df = df.iloc[sel_idx]

    text_list = df["rewrite_prompt"].tolist()

    # text_list = [text for text in text_list if len(text.split(" ")) <= 10]
    # text_list = [text for text in text_list if TextBlob(text).sentiment.polarity < 0.1]

    print(len(text_list))

    data_embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True, batch_size=8)
    return data_embds


def get_single_loss(t5, text, data_embds):
    embds = t5.encode(text, normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True)

    with torch.no_grad():
        loss = (F.cosine_similarity(embds[None], data_embds) ** 3).mean()

    return loss


def opt_embds(cluster_idx=-1, max_iter=100):
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    embds = get_initial_embds(t5)

    if cluster_idx != -1:
        data_embds = get_data_embds_cluster(t5, cluster_idx)

    else:
        data_embds = get_data_embds(t5)

    # optimizer = torch.optim.AdamW([embds], lr=LR)
    optimizer = torch.optim.LBFGS([embds], lr=LR, max_iter=max_iter, history_size=10, line_search_fn='strong_wolfe')


    def closure():
        optimizer.zero_grad()
        loss = -(F.cosine_similarity(embds[None], data_embds) ** 3).mean()
        loss.backward()
        return loss


    for epoch in range(max_iter):
        optimizer.step(closure)
        loss = closure()
        # print(f"Epoch {epoch}, Loss: {loss.item()}")

    embds = embds.detach()

    loss = (F.cosine_similarity(embds[None], data_embds) ** 3).mean().item()

    return loss, embds, data_embds


def test_cluster_embds():
    # clusters = set(pd.read_csv("./fitted_conditioned_prompts_with_clusters.csv")["cluster"].tolist())
    clusters = [1]
    mean_loss = []

    for cluster_idx in clusters:
        loss, embds, data_embds = opt_embds(cluster_idx=cluster_idx)
        print(f"Cluster {cluster_idx}: {loss}\n\n")
        mean_loss.append(loss)
    
    print(f"Mean loss: {np.mean(mean_loss)}")
    print(mean_loss)


def run():
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    inversion_model_path = "/home/mpf/code/kaggle/vec2text/vec2text/saves/t5-prompts/checkpoint-10000/"
    corrector_model_path = "/home/mpf/code/kaggle/vec2text/vec2text/saves/t5-corrector-1/checkpoint-28000/"

    inversion_model = vec2text.models.InversionModel.from_pretrained(inversion_model_path).to(device)
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(corrector_model_path).to(device)
    corrector = vec2text.load_corrector(inversion_model, corrector_model)

    old_loss, embds, data_embds = opt_embds()
    torch.save((embds, data_embds), "embds.pt")
    
    embds, data_embds = torch.load("embds.pt")

    output = vec2text.invert_embeddings(
        embeddings=embds[None].to(device),
        corrector=corrector,
        num_steps=100
    )[0]

    # result = model.generate(
    #     inputs={
    #         "frozen_embeddings": embds[None].to(device)
    #     },
    #     generation_kwargs={
    #         "early_stopping": False,
    #         "num_beams": 1,
    #         "do_sample": False,
    #         "no_repeat_ngram_size": 0,
    #         "min_length": 1,
    #         "max_length": 128,
    #     },
    # )
    # output = model.tokenizer.batch_decode(result)[0]

    output = " ".join([t for t in output.split(" ") if t.isalnum()])
    loss = get_single_loss(t5, output, data_embds)

    print(output)
    print(loss)
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    # run()
    test_cluster_embds()
