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


class ExLLamaModel:
    def __init__(self, model_dir: str, args={}):
        self.model_dir = model_dir
        self.args = DotDict(args) if args else self.get_default_args()
        self.args.model_dir = self.model_dir
        self.model, self.tokenizer, self.cache = self.load_exllama_model(self.args)

    def get_default_args(self):
        return DotDict(
            # {"model_dir": self.model_dir, "length": 8192, "tokens": 128, "max_input_len": 8192}
            #             {"model_dir": self.model_dir}
            {}
        )

    def warmup(self):
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        self.model.forward(input_ids, cache=None, input_mask=None, preprocess_only=True)

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

    def generate(self, prompt: str, max_tokens: int = 128, stop_token: int = -1):
        with torch.inference_mode():
            self.cache.current_seq_len = 0

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
                encode_special_tokens=True,
                stop_token=stop_token,
            )

        gen_text = output.replace(prompt, "").strip()

        return gen_text


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
