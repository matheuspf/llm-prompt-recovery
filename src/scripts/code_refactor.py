from src.generation.gpt_utils import call_gpt_api
from tqdm import tqdm
from pathlib import Path



def get_refactor_prompts(file_path: str):
    file_str = open(file_path, "r").read()

    system_prompt = "You are tasked to refactor python code to make it more modular and reusable without changing any functionality or behavior"
    prompt = \
f'''Given a python file, you will refactor the code to:

- Make it more usable
- Add typings (using python 3.10)
- Add simple docstrings
- Make the code more modular
- Clean up the code
- Turn relevant comments into conditional statements if that makes sense

Your output should be only the refactored code. Do not include the original code in your output.

I will parse your output exactly, so make sure it is only a Python script and nothing else.


Input python file:

python
```
{file_str}
```
'''
    return prompt, system_prompt


def refactor_file_gpt(file_path: str, out_path: str, model: str="gpt-4-turbo-preview"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt, system_prompt = get_refactor_prompts(file_path)

    result = call_gpt_api(prompt, model=model, system_prompt=system_prompt, temperature=0.2)
    result = result.replace("python\n```", "").replace("```python\n", "").strip()
    
    if result.endswith("```"):
        result = result[:-3].strip()
    
    with open(out_path, "w") as f:
        f.write(result)



if __name__ == "__main__":
    train_files = list(Path("src/train").glob("*.py"))
    gen_files = list(Path("src/generation").glob("*.py"))
    eval_files = list(Path("src/evaluation").glob("*.py"))
    file_paths = train_files + gen_files + eval_files
    
    for file_path in tqdm(file_paths):
        refactor_file_gpt(file_path, file_path)
