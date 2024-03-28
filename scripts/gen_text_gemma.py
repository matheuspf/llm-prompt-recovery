import time

import numpy as np
import pandas as pd
import torch
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..src.utils.data import *
from src.utils.exllama_utils import ExLLamaModel, Perplexity, get_gemma_prompt

np.random.seed(42)


def build_prompt(row, add_subject=True):
    prompt = f'''
Input text:
#####
{row["original_text"]}
#####

Rewritten text:
#####
{row["rewritten_text"]}
#####

'''

    if add_subject:
        # prompt += f'\n{row["subject"]}\n```\n\n'
        prompt += f'''Subject
######
{row["subject"]}
######


'''
    
    # else:
    #     prompt += 'What is the subject for the last example?'

    else:
        prompt += 'Subject\n######'
    
    return prompt


def get_fewshot_prompt(row, data_fs):
    fewshot_prompt = ""

    for idx, row_fs in data_fs.iterrows():
        fewshot_prompt += build_prompt(row_fs, add_subject=True)
    
    fewshot_prompt += build_prompt(row, add_subject=False)


#     prompt = f'''<start_of_turn>user
# {fewshot_prompt}<end_of_turn>
# <start_of_turn>model
# '''

    # prompt = fewshot_prompt



#     prompt = f'''Given a few examples of original and rewritten text, and their subjects, generate a subject for the last case.

# Examples:

# {fewshot_prompt}
# '''


#     prompt = f'''[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>
# Given a few examples of original and rewritten text, and their subjects, generate a subject for the last case.

# Examples:[/INST]

# {fewshot_prompt}
# '''

#     prompt = f'''<|im_start|>system
# You are an AI design to find the subject of a given text<|im_end|>
# <|im_start|>user
# Given a few examples of original and rewritten text, and their subjects, generate a subject for the last case.<|im_end|>
# <|im_start|>assistant
# {fewshot_prompt}
# '''


    prompt = f'''[INST] Given a few examples of original and rewritten text, and their subjects, generate a subject for the last case.
Examples:

[/INST] {fewshot_prompt}
'''

    return prompt


def get_embds_score(t5, pred, gt):
    pred_embds = t5.encode(pred, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    gt_embds = t5.encode(gt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)

    res = abs((cosine_similarity(gt_embds, pred_embds)) ** 3)

    return res[0][0]


# model_path = "/mnt/ssd/data/Gemma-7B-it-exl2"
# model_path = "/mnt/ssd/data/Gemma-7B-exl2"
# model_path = "/home/mpf/.cache/huggingface/hub/models--TheBloke--Mistral-7B-v0.1-GPTQ/snapshots/81de15eeac5938bc3b4065dfddf798fe5d215881/"
# model_path = "/home/mpf/.cache/huggingface/hub/models--TheBloke--Yi-34B-Chat-GPTQ/snapshots/8c21a20ec64957d64d386254b67b421573c8db04"
# model_path = "/home/mpf/.cache/huggingface/hub/models--TheBloke--Yi-34B-Chat-AWQ/snapshots/85ff7813de55a4324808d9443982b43afb97a316"
# model_path = "/mnt/ssd/data/Orion-14B-chat-exl2"
# model_path = "/home/mpf/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GPTQ/snapshots/ea078917a7e91c896787c73dba935f032ae658e9/"
model_path = "/home/mpf/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GPTQ/snapshots/7532d6bc89ef9300fb39d2d94ed4414ec534b72a/"
# model_path = "/mnt/ssd/data/Mistral-7B-Instruct-v0.2-exl2"


# data_fs = pd.read_csv("/kaggle/input/gemma_7b_gen/gemma_7b_exl2_gen.csv")
data_fs = pd.read_csv("/kaggle/input/gemma_7b_gen_v2/gemma_7b_exl2_gen.csv")

sel_ids = [idx for idx, row in data_fs.iterrows() if len(row["original_text"] + row["rewritten_text"]) < 1500][:20]
data_fs = data_fs.loc[sel_ids].copy()


data = load_data()["gemma_gen"]

subjects = []
scores = []

t5_name = "sentence-transformers/sentence-t5-base"
t5 = SentenceTransformer(t5_name, device="cuda")


model = ExLLamaModel(model_path)


with torch.inference_mode():
    for idx, row in tqdm(list(data.iterrows())[:100]):
        prompt = get_fewshot_prompt(row, data_fs)
        
        gen_text = model.generate(prompt, max_tokens=32)

        gen_text = gen_text.split("\n")[0].strip()
        gen_text = gen_text.replace("<|im_end|>", "").strip()
    
        pred_prompt = f'Please improve the following text rewriting it with the subject: "{gen_text}", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'

        score = get_embds_score(t5, pred_prompt, row["rewrite_prompt"])

        print(gen_text)
        print(row["rewrite_prompt"], "\n")
        print(score, "\n\n")

        subjects.append(gen_text)
        scores.append(score)


print(np.mean(scores))

