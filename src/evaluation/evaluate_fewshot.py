from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..exllama_utils import ExLLamaModel
from sentence_transformers import SentenceTransformer
from ..generation.prompts import gpt_conditioned_prompts_base


def get_embds_score(t5, pred, gt):
    pred_embds = t5.encode(pred, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)
    gt_embds = t5.encode(gt, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1)

    res = abs((cosine_similarity(gt_embds, pred_embds)) ** 3)

    return res[0][0]



class FewshotPrompt:
    def __init__(self, data_path, embds_model=None, embds_path=None, num_examples=10):
        self.data = self._load_fewshot_data(data_path)
        self.num_examples = num_examples
        self.embd_model = embds_model
        self.embds_original_text = None

        if embds_path is not None:
            self.embds_original_text = np.load(embds_path, allow_pickle=True)
    
    
#     def __call__(self, dct, embd=None):
#         fewshot_prompt = self.build_fewshot_prompt(dct, embd=embd)
#         pred_prompt = self.build_singleshot_prompt(dct, add_subject=False)

#         prompt = f'''[INST] Given a few examples of original and rewritten text, and their rewrite prompts, generate a rewrite prompt for the last case.
#     Examples:

#     [/INST] {fewshot_prompt}{pred_prompt}
#     '''
#         return prompt


#     def build_singleshot_prompt(self, row, add_subject=True):
#         prompt = f'''
# Input text:
# """"""
# {row["original_text"]}
# """"""

# Rewritten text:
# """"""
# {row["rewritten_text"]}
# """"""

# '''

#         if add_subject:
#             prompt += f'''Subject
# """"""
# {row["subject"]}
# """"""


# '''
#         else:
#             prompt += 'Subject\n""""""'
        
#         return prompt



    def __call__(self, dct, embd=None):
        fewshot_prompt = self.build_fewshot_prompt(dct, embd=embd)
        pred_prompt = self.build_singleshot_prompt(dct, add_subject=False)

        prompt = \
f'''[INST] Given a list of Original Text, Rewritten Text, and the most likely Prompt given by a user to generate the Rewritten Text, generate the most likely Prompt for the last case.

Given the task of understanding how text is rewritten by analyzing the Original Text and Rewritten Text, your goal is to deduce the specific instructions or prompt that was most likely used to generate the rewritten text from the original text. Consider the changes made in terms of style, tone, structure, and content. Assess whether the rewrite focuses on summarization, paraphrasing, stylistic alteration (e.g., formal to informal), or any specific content changes (e.g., making the text more concise, expanding on ideas, or altering the perspective). Follow this steps:

1. Read the Original Text: Start by thoroughly understanding the content, style, tone, and purpose of the original text. Note any key themes, technical terms, and the overall message.
2. Analyze the Rewritten Text: Examine how the rewritten text compares to the original. Identify what has been changed, added, or omitted. Pay close attention to changes in style (formal, informal), tone (serious, humorous), structure (paragraph order, sentence structure), and any shifts in perspective or emphasis.
3. Infer the Prompt: Based on your analysis, infer the most likely prompt that guided the rewriting process. Your inference should account for the observed changes in style, tone, structure, and content. Specify the type of task (e.g., summarize, paraphrase, make more accessible to a general audience), any specific directions evident from the changes, and any specific stylistic choice (e.g., 'in the S')

Based on your analysis return the prompt as if you were given the instruction your self like:
"Rewrite this text..."
"Transform this ... into ... based on the style of ..."

Make the prompt short and direct using a maximum of 20 words.
[/INST]

Examples:

[/INST] {fewshot_prompt}{pred_prompt}'''

        return prompt


    def build_singleshot_prompt(self, row, add_subject=True):
        prompt = \
f'''Original Text
""""""
{row["original_text"]}
""""""

Rewritten text:
""""""
{row["rewritten_text"]}
""""""

Prompt:
""""""
'''
        if add_subject:
            prompt += \
f'''{row["rewrite_prompt"]}
""""""
------------------------------------------------------------------------------------------------


'''
        return prompt
        
    
    
    def build_fewshot_prompt(self, dct, embd=None):
        fewshot_examples = self.get_fewshot_examples(dct, embd=embd)
        fewshot_prompts = [self.build_singleshot_prompt(dct_fs) for idx, dct_fs in fewshot_examples.iterrows()]
        # pred_prompt = self.build_singleshot_prompt(dct, add_subject=False)
        # fewshot_prompt = "\n".join(fewshot_prompts + [pred_prompt])
        fewshot_prompt = "\n".join(fewshot_prompts)

        return fewshot_prompt
    
        
    def get_fewshot_examples(self, dct, embd=None):
        return self.data.head(self.num_examples)

        sims = cosine_similarity(embd[None, ...], self.embds_original_text)[0]
        idxs = np.argsort(-sims)[:self.num_examples]#[::-1]
        data = self.data.iloc[idxs].copy()
        return data

    
    def _load_fewshot_data(self, data_path):
        data = pd.read_csv(data_path).reset_index(drop=True)
        # sel_idx = [idx for idx, dct in data.iterrows() if len(dct["rewritten_text"]) >= 300 and len(dct["rewritten_text"]) <= 1500]
        # data = data.iloc[sel_idx].reset_index(drop=True)
        return data



def get_dataset(data_path="/kaggle/input/df_with_emb.parquet", num_examples=None):
    parquet = pd.read_parquet(data_path)

    if num_examples is not None:
        # parquet = parquet.head(num_examples)
        parquet = parquet.sample(num_examples, random_state=42)

    embds_dict = {}

    for c in ("original_text", "rewrite_prompt", "rewritten_text"):
        embds_dict[c] = np.array(parquet[[f"{c}_emb_{i}" for i in range(768)]].values.tolist())
    
    df = parquet[["original_text", "rewrite_prompt", "rewritten_text"]].reset_index(drop=True)

    df = df[df["original_text"].apply(lambda x: len(x) >= 300 and len(x) <= 2000)].reset_index(drop=True)
    df = df[df["rewritten_text"].apply(lambda x: len(x) >= 200 and len(x) <= 3000)].reset_index(drop=True)

    return df, embds_dict


df, embds_dict = get_dataset(num_examples=5000)


embds_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# new_embds = embds_model.encode(df["original_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
# new_embds = embds_model.encode(df["rewritten_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)


# prompter = FewshotPrompt(data_path="/kaggle/input/gemma_7b_gen_v2/gemma_7b_exl2_gen.csv", num_examples=10)
# prompter = FewshotPrompt(data_path="/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_20240324_193550.csv", num_examples=10)
prompter = FewshotPrompt(
    data_path="/kaggle/input/gemma_rewritten_text_exllama/proc_dataset_20240324_193550.csv",
    embds_path="/kaggle/input/gemma_rewritten_text_exllama/embds.npy",
    num_examples=10
)

model = ExLLamaModel("/mnt/ssd/data/Mistral-7B-Instruct-v0.2-exl2")
# model = ExLLamaModel("/mnt/ssd/data/Smaug-34B-v0.1-4.0bpw-h6-exl2", args={"length": 4096, "max_input_len": 4096})

t5 = SentenceTransformer("sentence-transformers/sentence-t5-base")

scores = []


for idx, row in tqdm(df.iterrows(), total=len(df)):
    # prompt = prompter(row, new_embds[idx])
    prompt = prompter(row)

    gen_text = model.generate(prompt, max_tokens=32)
    gen_text = gen_text.replace('"', '').strip()
    gen_text = gen_text.split("\n")[0].strip()

    pred_prompt = f'Please improve the following text rewriting it with the subject: "{gen_text}", using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'Please improve the following text "{gen_text}" using the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'
    # pred_prompt = f'{gen_text}. Use the writing style of , maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style. Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.'

    score = get_embds_score(t5, pred_prompt, row["rewrite_prompt"])
    scores.append(score)

    # print(prompt)
    # print(len(prompt))
    print(gen_text)
    print(row["rewrite_prompt"])
    print(score, "\n\n")
    print(np.mean(scores))
    # import pdb; pdb.set_trace()


scores = np.array(scores)

print(np.mean(scores))

import pdb; pdb.set_trace()


## ORG 0.5574

# NEW prompt: 0.5803

