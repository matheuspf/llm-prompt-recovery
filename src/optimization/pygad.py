import json
import copy
import pygad
import numpy as np
from .mean_prompt_tokens import get_top_words
from .data import *
from sklearn.metrics.pairwise import cosine_similarity


def get_token_map():
    df = get_dataset_pedro_lowercase()
    text_list = df["rewrite_prompt"].tolist()
    all_words = get_top_words(text_list, None, None)

    token_map = np.array(all_words)
    inv_token_map = {word: i for i, word in enumerate(all_words)}

    return token_map, inv_token_map


def get_embds(t5):
    df = get_dataset_pedro()
    text_list = df["rewrite_prompt"].tolist()
    
    embds = t5.encode(text_list, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    return embds


t5 = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)
token_map, inv_token_map = get_token_map()
embds = get_embds(t5)


def get_initial_population(num_examples=5):
    prompt_dict = json.load(open("/home/mpf/code/kaggle/llm-prompt/src/optimization/mean_prompt_sel_tokens_updated_alpha_692_clean.json"))
    prompts = [dct["text"].split(" ") for dct in prompt_dict]

    prompt = sorted(prompts, key=lambda x: len(x), reverse=True)[0]
    new_prompts = [prompt]

    for _ in range(num_examples):
        new_prompt = copy.deepcopy(prompt)
        np.random.shuffle(new_prompt)
        new_prompts.append(new_prompt)
    
    new_prompts_ids = [[inv_token_map[word] for word in prompt] for prompt in new_prompts]

    return new_prompts_ids



def fitness_function(ga_instance, solution, solution_idx):
    text = " ".join(token_map[solution])
    text_embd = t5.encode(text, normalize_embeddings=True, show_progress_bar=False)
    score = (cosine_similarity(embds, text_embd.reshape(1, -1)) ** 3).mean()

    # print(score, text)

    return score


def crossover_func(parents, offspring_size, ga_instance):
    return np.zeros(offspring_size)


last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}")
    print("-"*100, "\n")
    last_fitness = ga_instance.best_solution()[1]


ga_instance = pygad.GA(
    gene_type=np.int32,
    num_generations=100,
    num_parents_mating=5,
    mutation_probability=0.5,
    fitness_func=fitness_function,
    sol_per_pop=None,
    num_genes=None,
    initial_population=get_initial_population(),
    on_generation=callback_generation
)

ga_instance.run()
