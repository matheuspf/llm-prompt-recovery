import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
                    module.num_embeddings, module.embedding_dim, _weight=module.weight
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
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    token=token,
)


replace_embedding(model)

prompt_template = "{prompt}\n{input_text}"

prompt = "Convert this into a sea shanty:"
input_text = "The competition dataset comprises text passages that have been rewritten by the Gemma LLM according to some rewrite_prompt instruction. The goal of the competition is to determine what prompt was used to rewrite each original text.  Please note that this is a Code Competition. When your submission is scored, this example test data will be replaced with the full test set. Expect roughly 2,000 original texts in the test set."
output_text = "Here is your shanty: (Verse 1) The text is rewritten, the LLM has spun, With prompts so clever, they've been outrun. The goal is to find, the prompt so bright, To crack the code, and shine the light. (Chorus) Oh, this is a code competition, my dear, With text and prompts, we'll compete. Two thousand texts, a challenge grand, To guess the prompts, hand over hand.(Verse 2) The original text, a treasure lost, The rewrite prompt, a secret to be"

chat = [
    {"role": "user", "content": prompt_template.format(prompt=prompt, input_text=input_text)},
    {"role": "model", "content": output_text},
]
text = tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)

if text.endswith("<start_of_turn>model\n"):
    text = text[: -len("<start_of_turn>model\n")]

prompt_end = text.find(prompt) + len(prompt) + 1

prompt = text[:prompt_end]
model_text = text[prompt_end:]

prompt_ids = tokenizer(
    prompt[:25] + "ABC DEF ABC DEF ABC DEF ABC DEF ABC", return_tensors="pt"
).to(device)["input_ids"]
model_ids = tokenizer(model_text, return_tensors="pt").to(device)["input_ids"][:, 1:]
tgt_ids = tokenizer(text, return_tensors="pt").to(device)["input_ids"]

prompt_embds = model.model.embed_tokens(prompt_ids).clone().detach().requires_grad_(True)
# prompt_embds = torch.zeros(1, 14, 3072, device=device, dtype=torch.bfloat16, requires_grad=True)
model_embds = model.model.embed_tokens(model_ids).detach()

with torch.no_grad():
    tgt_embds = model(input_ids=tgt_ids, return_dict=True)["logits"]
    # tgt_labels = tgt_embds.argmax(-1)

optimizer = optim.AdamW([prompt_embds], lr=1e-2)
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()


for i in range(10):
    optimizer.zero_grad()
    input_embds = torch.cat((prompt_embds, model_embds), 1)

    tokens = (model.model.embed_tokens.embedding.weight @ input_embds[0].T).argmax(0)
    print(tokens)
    exit()

    pred_embds = model(input_ids=input_embds, return_dict=True)["logits"]
    # loss = loss_fn(pred_embds, tgt_labels)
    loss = loss_fn(pred_embds, tgt_embds)

    loss.backward()
    optimizer.step()

    # Unmap the embeddings
    import pdb

    pdb.set_trace()
    tokens = (model.model.embed_tokens.embedding.weight @ input_embds[0].T).argmax(0)

    print(loss.item())
