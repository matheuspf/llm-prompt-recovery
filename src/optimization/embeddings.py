import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from .data import *
import vec2text


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


def get_single_loss(t5, text, data_embds):
    embds = t5.encode(text, normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True)

    with torch.no_grad():
        loss = -(F.cosine_similarity(embds[None], data_embds) ** 3).mean()

    return loss


def opt_embds(max_iter=2):
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    embds = get_initial_embds(t5)
    data_embds = get_data_embds(t5)

    # optimizer = torch.optim.AdamW([embds], lr=LR)
    optimizer = torch.optim.LBFGS([embds], lr=LR, max_iter=max_iter, history_size=10, line_search_fn='strong_wolfe')


    def closure():
        optimizer.zero_grad()
        loss = -(F.cosine_similarity(embds[None], data_embds) ** 3).mean()
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

    embds = embds.detach()

    return embds, data_embds


def run():
    t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

    model_path = "/home/mpf/code/kaggle/vec2text/vec2text/saves/t5-prompts/checkpoint-6000/"
    model = vec2text.models.InversionModel.from_pretrained(model_path).to(device)

    # embds, data_embds = opt_embds()
    # torch.save((embds, data_embds), "embds.pt")
    
    embds, data_embds = torch.load("embds.pt")

    result = model.generate(
        inputs={
            "frozen_embeddings": embds[None].to(device)
        },
        generation_kwargs={
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            "min_length": 1,
            "max_length": 128,
        },
    )

    output = model.tokenizer.batch_decode(result)[0]
    loss = get_single_loss(t5, output, data_embds)

    print(loss)
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    run()


