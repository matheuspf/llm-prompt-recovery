import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchsort
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GemmaForCausalLM

from src.utils.exllama_utils import Perplexity


class CustomEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        if x.dtype != torch.long:
            return x

        return self.embedding(x)


def replace_embedding(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Embedding):
            setattr(
                model,
                name,
                CustomEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    _weight=module.weight,
                    padding_idx=module.padding_idx,
                ),
            )
        else:
            replace_embedding(module)


model_name = "google/gemma-7b-it"
device = "cuda:0"

token = open(os.path.expanduser("~/.hugginface_token")).read().strip()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.float16,
    # quantization_config=quantization_config,
    # low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    token=token,
)
model = model.eval().to(device)
model.requires_grad_(False)


replace_embedding(model)

prompt_template = "{prompt}\n{input_text}"

text = """<start_of_turn>user
Convert this into a sea shanty:
The competition dataset comprises text passages that have been rewritten by the Gemma LLM according to some rewrite_prompt instruction. The goal of the competition is to determine what prompt was used to rewrite each original text.  Please note that this is a Code Competition. When your submission is scored, this example test data will be replaced with the full test set. Expect roughly 2,000 original texts in the test set.<end_of_turn>
<start_of_turn>model
Here is your shanty: (Verse 1) The text is rewritten, the LLM has spun, With prompts so clever, they've been outrun. The goal is to find, the prompt so bright, To crack the code, and shine the light. (Chorus) Oh, this is a code competition, my dear, With text and prompts, we'll compete. Two thousand texts, a challenge grand, To guess the prompts, hand over hand.(Verse 2) The original text, a treasure lost, The rewrite prompt, a secret to be"""

target_tokens = tokenizer(text, return_tensors="pt").to(device)["input_ids"][0]
special_tokens = torch.tensor(
    list(tokenizer.added_tokens_decoder.keys()), device=device, dtype=torch.long
)
special_tokens_mask = (target_tokens == special_tokens[:, None]).any(0)

end_of_prompt = "<start_of_turn>model\n"
prompt_end_pos = text.find(end_of_prompt) + len(end_of_prompt) + 1
prompt_tokens = tokenizer(text[:prompt_end_pos], return_tensors="pt").to(device)["input_ids"][0]

start_of_input = "<start_of_turn>user\n"
end_of_input = "Convert this into a sea shanty:\n"

input_start_pos = text.find(start_of_input) + len(start_of_input)
input_end_pos = text.find(end_of_input) + len(end_of_input)

bos_tokens = tokenizer(text[:input_start_pos], return_tensors="pt").to(device)["input_ids"][0]
# input_tokens = tokenizer(text[input_start_pos:input_end_pos], return_tensors="pt").to(device)["input_ids"][0][1:]
input_tokens = tokenizer("Convert this into a xxx shanty:\n", return_tensors="pt").to(device)[
    "input_ids"
][0][1:]
# input_tokens = tokenizer("ABC DEF GHI KLM ABC DEF GHI", return_tensors="pt").to(device)["input_ids"][0][1:]
model_tokens = tokenizer(text[input_end_pos:], return_tensors="pt").to(device)["input_ids"][0][1:]

# target_tokens[special_tokens_mask] = -100
# target_tokens[:len(prompt_tokens) - 1] = -100
# target_tokens[len(bos_tokens) + len(input_tokens):] = -100
# target_tokens[:len(bos_tokens) + len(input_tokens)] = -100
# target_tokens[:len(bos_tokens)] = -100
# target_tokens = target_tokens.unsqueeze(0)


input_pos = (
    torch.nn.functional.one_hot(input_tokens, len(tokenizer))
    .to(torch.float16)
    .requires_grad_(True)
)


bos_embds = model.model.embed_tokens(bos_tokens).clone().detach()
# input_embds = model.model.embed_tokens(input_tokens).clone().detach().requires_grad_(True)
input_embds = model.model.embed_tokens(input_tokens).clone().detach()
model_embds = model.model.embed_tokens(model_tokens).clone().detach()

embds = torch.cat((bos_embds, input_embds, model_embds), 0)

input_org_embds = input_embds.clone().detach()
org_embds = embds.clone().detach()

closest = torch.argmin(torch.cdist(input_embds, model.model.embed_tokens.embedding.weight), dim=1)

loss_fn = Perplexity()
optimizer = optim.AdamW([input_pos], lr=1e-3, weight_decay=1e-3)
# optimizer = optim.LBFGS([input_embds], lr=1e-1, max_iter=10, history_size=100)
# optimizer = optim.LBFGS([input_pos], lr=1e-1, max_iter=10, history_size=100)


# pred_tokens = torch.cdist(torch.cat((bos_embds, input_embds), 0), model.model.embed_tokens.embedding.weight).argmin(-1)
# pred_text = tokenizer.decode(pred_tokens)
# print(pred_text, "\n\n")


def closure():
    optimizer.zero_grad()

    # input_embds = model.model.embed_tokens(input_pos.argmax(-1))
    # input_embds = model.model.embed_tokens.embedding.weight[input_pos.argmax(-1)]

    input_embds = (input_pos * 1.0).softmax(1) @ model.model.embed_tokens.embedding.weight

    # input_embds = torchsort.soft_rank(input_pos) @ model.model.embed_tokens.embedding.weight

    embds = torch.cat((bos_embds, input_embds, model_embds), 0)

    pred = model(input_ids=embds.unsqueeze(0), return_dict=True)["logits"]

    pred_pos = pred[:, len(bos_tokens) + len(input_tokens) :]
    target_tokens_pos = target_tokens[len(bos_tokens) + len(input_tokens) :].unsqueeze(0)

    loss = loss_fn(pred_pos, target_tokens_pos)
    # loss += input_embds.norm() * 1e-1
    # loss += (input_embds.norm() - input_org_embds.norm()).pow(2) * 1e-1

    # loss += (embds.norm() - org_embds.norm()).pow(2) * 1e-2
    loss.backward()

    return loss


for i in range(1000):
    # loss = optimizer.step(closure)

    optimizer.zero_grad()

    input_embds = (input_pos * 1e3).softmax(1) @ model.model.embed_tokens.embedding.weight
    embds = torch.cat((bos_embds, input_embds, model_embds), 0)

    # tokens = torch.cdist(embds, model.model.embed_tokens.embedding.weight).argmin(-1)
    # embds = model.model.embed_tokens(tokens)

    pred = model(input_ids=embds.unsqueeze(0), return_dict=True)["logits"]
    # loss = loss_fn(pred, target_tokens) + (input_embds.norm() - input_org_embds.norm()).pow(2) * 1e-3

    pred_pos = pred[:, len(bos_tokens) + len(input_tokens) :]
    target_tokens_pos = target_tokens[len(bos_tokens) + len(input_tokens) :].unsqueeze(0)

    loss = loss_fn(pred_pos, target_tokens_pos)
    loss.backward()

    print(input_pos.grad)

    optimizer.step()

    # if i % 20 == 0:
    if 1:
        with torch.no_grad():
            # pred_tokens = torch.cdist(torch.cat((bos_embds, input_embds), 0), model.model.embed_tokens.embedding.weight).argmin(-1)
            # pred_text = tokenizer.decode(pred_tokens)

            pred_text = tokenizer.decode(input_pos.argmax(-1))

            # print(input_embds.norm().item())
            print(loss.item())
            print(pred_text)
            print("\n\n")

    # rec_tokens = torch.cdist(input_embds, model.model.embed_tokens.embedding.weight).argmin(-1)
    # input_embds = model.model.embed_tokens(rec_tokens).clone().detach().requires_grad_(True)
    # optimizer = optim.AdamW([input_embds], lr=1e-1, weight_decay=1e-3)
