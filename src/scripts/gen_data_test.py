import json
import os
import time
from pathlib import Path

import tiktoken
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_api(
    prompt,
    function_dct=None,
    model="gpt-3.5-turbo-0613",
    system_prompt="You are an AI designed to assist with a request",
    temperature=0.5,
    retries=3,
    retry_wait=2,
):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    encoding = tiktoken.encoding_for_model(model)
    # num_tokens = len(encoding.encode(prompt)) + len(encoding.encode(str(function_dct)))

    for idx in range(retries):
        # try:
        if function_dct is None:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=4096
            )
            result = response.choices[0].message.content

        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                functions=[function_dct],
                function_call={"name": function_dct["name"]},
                temperature=temperature,
                max_tokens=4096,
            )
            msg = response.choices[0].message
            result = msg.function_call.arguments
            result = json.loads(result)

        return result

        # except:
        #     if idx == retries - 1:
        #         print("Error:\n\n", prompt)

        #     else:
        #         print(f"Retry {idx+1}")
        #         time.sleep(retry_wait)

        #     continue

    return None


# result = call_api("How to use OpenAI API for making function calling requests using GPT-4 for a List[tuple[str, str]] expected return type? Code only in Python:\npython\n```", model="gpt-4-turbo-preview", temperature=0.5)

# print(result)


system_prompt = "You are tasked to design prompts for a LLM"


def create_prompt(dct):
    prompt = """You are going to generate a list of 20 prompts that will be given as input for a LLM.
The prompts will be used to generate rewrite tasks for a language model, given an input text to be rewritten.

For each generated prompt also specify some subjects of the rewrite task, the main idea of it - some words or short phrases contained inside the prompt.

Some examples:

json
```
{
    "prompts": [
        {
            "prompt": "Improve this text",
            "subject": "text improvement"
        },
        {
            "prompt": "Please make this text sound more formal for a business email, where I am asking for a raise",
            "subject": "business email, sound formal, asking for a raise"
        },
        {
            "prompt": "Convert this into a sea shanty",
            "subject": "sea shanty"
        },
        {
            "prompt": "Please improve the following text using the writing style of Alice in Wonderland, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style.",
            "subject": "Alice in Wonderland, mantain original meaning, alter tone"
        },
        {
            "prompt": "Make this text sound more formal",
            "subject": "sound formal"
        },
        {
            "prompt": "Change the second word to foo and the last word to bar",
            "subject": "word replacement, foo, bar"
        }
    ]
}
```
"""

    function_dct = {
        "name": "generate_rewrite_prompt",
        "description": "Generate rewrite prompts for a LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "description": "A list of size 20 containing dictionaries with the prompt and the subject of the rewrite task",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt representing the rewrite task",
                            },
                            "subject": {
                                "type": "string",
                                "description": "The subject of the rewrite task, the main idea of it - some words or a short phrases contained inside the prompt",
                            },
                        },
                    },
                },
            },
            "required": ["question", "answer", "sentence_ids", "options"],
        },
    }
    return prompt, function_dct


prompt, function_dct = create_prompt({})


result = call_api(
    prompt, function_dct, system_prompt=system_prompt, model="gpt-4-turbo-preview", temperature=0.5
)

time_id = time.strftime("%Y%m%d_%H%M%S")

# out_path = Path(f"/kaggle/input/gen_data_test/test_{time_id}.json")
out_path = Path(f"/kaggle/input/gen_data_test_v2/test_{time_id}.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(result, f, indent=4)

print(json.dumps(result, indent=4))
