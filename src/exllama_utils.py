import os
import time
from threading import Lock

import torch
from exllamav2 import ExLlamaV2Cache, model_init
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# _lock = Lock()
_tokenizer_cache = {}


def get_gemma_prompt(
    chat: str | list[dict],
    model_name: str = "gg-hf/gemma-7b-it",
    token_path: str = "~/.hugginface_token",
):
    global _tokenizer_cache

    if isinstance(chat, str):
        chat = [{"role": "user", "content": chat}]

    for c in chat:
        assert (
            "role" in c and "content" in c
        ), "Chat must be a list of dicts with 'role' and 'content' keys."

    # with _lock:
    if model_name not in _tokenizer_cache:
        token = open(os.path.expanduser(token_path)).read().strip()
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, token=token)

    tokenizer = _tokenizer_cache[model_name]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt = prompt.replace("<bos>", "")

    return prompt


class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        # perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity


class DotDict(dict):
    """dot.notation access to dictionary attributes with None for missing keys."""

    def __getattr__(self, item):
        return self.get(item, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ExLLamaModel:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.args = self.get_default_args()
        self.model, self.tokenizer, self.cache = self.load_exllama_model(self.args)

    def get_default_args(self):
        return DotDict(
            # {"model_dir": self.model_dir, "length": 8192, "tokens": 128, "max_input_len": 8192}
            {"model_dir": self.model_dir}
            # {"model_dir": self.model_dir, "length": 4096, "tokens": 128, "max_input_len": 4096}
        )

    def get_generation_settings(self):
        settings = ExLlamaV2Sampler.Settings()
        # settings.temperature = 0.75
        # settings.top_k = 100
        # settings.top_p = 0.75
        # settings.token_repetition_penalty = 1.05

        settings.temperature = 0.0
        settings.top_k = 1
        settings.top_p = 1.0
        settings.token_repetition_penalty = 1.0

        # settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        return settings

    def load_exllama_model(self, args):
        model, tokenizer = model_init.init(args, allow_auto_split=True)
        cache = None

        cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(cache)

        return model, tokenizer, cache

    def perplexity(self, prompt: str):
        perp_fn = Perplexity()

        with torch.inference_mode():
            ids = self.tokenizer.encode(prompt, add_bos=True)

            logits = self.model.forward(ids, cache=None)

            ids = ids.to(logits.device)

            perp = perp_fn(logits, ids).item()

        return perp

    def generate(self, prompt: str, max_tokens: int = 128):
        self.cache.current_seq_len = 0
        
        with torch.inference_mode():
            generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)

            if not self.args.no_warmup:
                generator.warmup()

            settings = self.get_generation_settings()

            output = generator.generate_simple(
                prompt,
                settings,
                num_tokens=max_tokens,
                token_healing=True,
                add_bos=True,
                encode_special_tokens=True
            )
            
            gen_text = output.replace(prompt, "").strip()

            torch.cuda.synchronize()


        return gen_text


if __name__ == "__main__":
    model = ExLLamaModel("/mnt/ssd/data/Gemma-7B-it-exl2")
    # model = ExLLamaModel("/home/mpf/.cache/huggingface/hub/models--google--gemma-7b-it/snapshots/b078e50c64458cf42df13eb97e368c00ce3a7b00/")
    # model = ExLLamaModel("/home/mpf/.cache/huggingface/hub/models--TechxGenus--gemma-7b-it-GPTQ/snapshots/8f760fc349e7e835a810a3b8b860bc3e27b9fbaa/")

    prompt = """<start_of_turn>user
    Write me a poem about Machine Learning<end_of_turn>
    <start_of_turn>model
    In the realm of data, a tale unfolds,
    Where algorithms dance, stories untold.
    Machine learning, a force unleashed,
    Uncovers patterns, insights, and peace.
    
    Data whispers secrets, hidden in the past,
    Machine learning listens, never too fast.
    It learns from examples, patterns emerge,
    And makes predictions, with an agile surge.
    
    From simple tasks to complex ones, it takes flight,
    With the power of learning, shining light.
    It paints a canvas, with data's hue,
    And brings us closer, to what we knew."""

    prompt = 'Convert this into a sea shanty: """"""The competition dataset comprises text passages that have been rewritten by the Gemma LLM according to some rewrite_prompt instruction. The goal of the competition is to determine what prompt was used to rewrite each original text.  Please note that this is a Code Competition. When your submission is scored, this example test data will be replaced with the full test set. Expect roughly 2,000 original texts in the test set."""""""""'

    prompt = get_gemma_prompt(prompt)

    output = model.generate(prompt)
    # output = model.perplexity(prompt)

    print(output)
