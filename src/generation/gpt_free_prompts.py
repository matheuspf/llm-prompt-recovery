import json
from tqdm import tqdm
import copy
import numpy as np
import time
from pathlib import Path
from .gpt_utils import call_gpt_api
from .prompts import gpt_free_prompts_base


np.random.seed(27001)


system_prompt = "You are tasked to design text rewrite prompts for a LLM"

def build_prompt(content):
    prompt = \
f'''Generate a parseable json list of possible rewrite prompts a person may ask a LLM to do given an input text. The generated json will be in the format:

json
```
[
  {{
    "prompt": "",
    "subject": ""
  }},
  ...
  ...
]
```

Where:

- `prompt` is the rewrite prompt that a person may ask a LLM to do given an input text
- `subject` is the subject of the rewrite prompt, only one or a few words describing the rewrite task, the main idea of it

The prompts must be different from each other, trying to be diverse and cover a wide range of possible rewrite tasks.

Make sure to output only a json list, I will parse it.

I will give some examples and you will generate new ones that differ from the examples and from each other.


Examples:

json
```
{json.dumps(content, indent=2)}
```

Begin!
'''

    return prompt



def get_results(num_fewshot=5, max_runs=10, api_params={}):
    content = copy.deepcopy(gpt_free_prompts_base)
    
    np.random.shuffle(content)
    content = content[:num_fewshot]

    for run_idx in tqdm(range(max_runs)):
        prompt = build_prompt(content)
        # result = call_gpt_api(prompt, system_prompt=system_prompt, model="gpt-3.5-turbo", temperature=0.5)
        result = call_gpt_api(prompt, **api_params)
        
        try:
            result = json.loads(result.replace("json", "").replace("```", ""))
        except:
            print("INVALID JSON:\n", result, "\n")
            continue
        
        content += result
    
    content = content[num_fewshot:]

    return content


def get_prompts(num_iters=10, num_fewshot=5, max_runs=10, api_params={}):
    all_results = []
    
    for iter_idx in tqdm(range(num_iters)):
        all_results += get_results(num_fewshot=num_fewshot, max_runs=max_runs, api_params=api_params)
    
    all_results = gpt_free_prompts_base + all_results

    return all_results


api_params = {"system_prompt": system_prompt, "model": "gpt-3.5-turbo", "temperature": 0.5}

result = get_prompts(num_iters=5, num_fewshot=5, max_runs=10, api_params=api_params)

time_id = time.strftime("%Y%m%d_%H%M%S")

out_path = Path(f"/kaggle/input/gpt_free_prompts/{time_id}.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(result, f, indent=4)

# print(json.dumps(result, indent=4))

print(len(result))



