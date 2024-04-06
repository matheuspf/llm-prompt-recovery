import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from .data import *

device = "cuda:0"
LR = 1e-3
EPOCHS = 10
DISP_STEPS = 1


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


t5 = SentenceTransformer("sentence-transformers/sentence-t5-base")
embds = get_initial_embds(t5)
data_embds = get_data_embds(t5)

# optimizer = torch.optim.AdamW([embds], lr=LR)
optimizer = torch.optim.LBFGS([embds], lr=LR, max_iter=10, history_size=10, line_search_fn='strong_wolfe')


def closure():
    optimizer.zero_grad()
    loss = -(F.cosine_similarity(embds.unsqueeze(0), data_embds) ** 3).mean()
    loss.backward()
    return loss


for epoch in range(EPOCHS):
    # optimizer.zero_grad()
    # loss = -(F.cosine_similarity(embds.unsqueeze(0), data_embds) ** 3).mean()
    # loss.backward()
    # optimizer.step()

    optimizer.step(closure)
    loss = closure()
    
    if epoch % DISP_STEPS == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

