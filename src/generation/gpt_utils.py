import json
import os
import time
from pathlib import Path

import tiktoken
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_gpt_api(
    prompt,
    function_dct=None,
    model="gpt-3.5-turbo",
    system_prompt="You are an AI designed to assist with a request",
    temperature=0.5,
    retries=3,
    retry_wait=2,
):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    for idx in range(retries):
        try:
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

        except:
            if idx == retries - 1:
                print("Error:\n\n", prompt)

            else:
                print(f"Retry {idx+1}")
                time.sleep(retry_wait)

            continue

    return None
