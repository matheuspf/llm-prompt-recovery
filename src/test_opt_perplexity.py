import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchsort
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GemmaForCausalLM

from exllama_utils import Perplexity


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

start_of_input = "<start_of_turn>user\nConvert this into a"
end_of_input = "Convert this into a sea"

input_start_pos = text.find(start_of_input) + len(start_of_input)
input_end_pos = text.find(end_of_input) + len(end_of_input)

bos_tokens = tokenizer(text[:input_start_pos], return_tensors="pt").to(device)["input_ids"][0]

# input_tokens = tokenizer(text[input_start_pos:input_end_pos], return_tensors="pt").to(device)["input_ids"][0][1:]
input_tokens = tokenizer("xxx", return_tensors="pt").to(device)["input_ids"][0][1:]
# input_tokens = tokenizer("ABC DEF GHI KLM ABC DEF GHI", return_tensors="pt").to(device)["input_ids"][0][1:]

model_tokens = tokenizer(text[input_end_pos:], return_tensors="pt").to(device)["input_ids"][0][1:]

bos_embds = model.model.embed_tokens(bos_tokens).clone().detach()
input_embds = model.model.embed_tokens(input_tokens).clone().detach().requires_grad_(True)
model_embds = model.model.embed_tokens(model_tokens).clone().detach()

loss_fn = Perplexity()

# optimizer = optim.AdamW([input_embds], lr=1e-1)

# with torch.no_grad():
#     embds = torch.cat((bos_embds, input_embds, model_embds), 0)
#     pred_tokens = torch.cdist(embds, model.model.embed_tokens.embedding.weight).argmin(-1)
#     pred_text = tokenizer.decode(pred_tokens)
#     print(pred_text, "\n\n")

#     bos_embds /= bos_embds.norm(dim=1).unsqueeze(1)
#     model_embds /= model_embds.norm(dim=1).unsqueeze(1)

#     embds_norm = embds.norm(dim=1)


for i in range(1000):
    # optimizer.zero_grad()

    embds = torch.cat((bos_embds, input_embds, model_embds), 0)

    pred = model(input_ids=embds.unsqueeze(0), return_dict=True)["logits"]

    pred_pos = pred[:, len(bos_tokens) + len(input_tokens) :]
    target_tokens_pos = target_tokens[len(bos_tokens) + len(input_tokens) :].unsqueeze(0)

    loss = loss_fn(pred_pos, target_tokens_pos)

    # if input_embds.norm().item() < 3 or input_embds.norm().item() > 8:
    #     loss += 1e-3 * (input_embds.norm() - 5) ** 2

    loss.backward()
    input_embds = (input_embds - 1e-1 * input_embds.grad).clone().detach().requires_grad_(True)

    # optimizer.step()

    # if i % 20 == 0:
    if 1:
        with torch.no_grad():
            pred_tokens = torch.cdist(
                input_embds, model.model.embed_tokens.embedding.weight
            ).argmin(-1)
            pred_text = tokenizer.decode(pred_tokens)

            print(input_embds.norm().item())
            print(loss.item())
            print(pred_text)
            print("\n\n")

    # rec_tokens = torch.cdist(input_embds, model.model.embed_tokens.embedding.weight).argmin(-1)
    # input_embds = model.model.embed_tokens(rec_tokens).clone().detach().requires_grad_(True)
    # optimizer = optim.AdamW([input_embds], lr=1e-1, weight_decay=1e-3)
