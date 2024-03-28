import json
import os

import tiktoken
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_api(
    prompt,
    function_dct=None,
    model="gpt-3.5-turbo-0613",
    system_msg="You are an AI designed to assist with a request",
    temperature=0.5,
    retries=3,
    retry_wait=2,
):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]

    encoding = tiktoken.encoding_for_model(model)
    # num_tokens = len(encoding.encode(prompt)) + len(encoding.encode(str(function_dct)))

    for idx in range(retries):
        try:
            if function_dct is None:
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=4096
                )
                import pdb

                pdb.set_trace()
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
                result = msg.to_dict()["function_call"]["arguments"]
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


t1_py = open("t1.py").read()
t1_xml = open("t1.xml").read()
t2_xml = open("t2.xml").read()


prompt = f"""You will be given one XML snippet and the Python code used to generate it.
After that, one more XML snippet will be provided for which you will generate the Python code for it.

Use the exact same functions and names as the Python example. Make the code generic as to accept other inputs for the `classes` variable.

Only output Python code, I will parse it:

xml
```
{t1_xml}
```

python
```
{t1_py}
```

====================================================================================================

xml
```
{t2_xml}
```

python
```
"""


result = call_api(
    prompt,
    model="gpt-4-turbo-preview",
    system_msg="You are an AI designed to generate Python code from XML snippets",
    temperature=0.1,
)


# def create_prompt(dct):
#     prompt = f"""
# ```
# Title: {dct["title"]}

# Context:

# {dct["selected_sentences"]}
# ```
# """
#     function_dct = {
#         "name": "generate_question",
#         "description": "Generate a question, answer and possible options for a given Wikipedia text snippet.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "question": {
#                     "type": "string",
#                     "description": "A detailed question about the text relating two or more sentences",
#                 },
#                 "answer": {
#                     "type": "string",
#                     "description": "The answer to the question spanning two or more sentences",
#                 },
#                 "sentence_ids": {
#                     "type": "array",
#                     "description": "A list of numbers representing the relevant sentences with respect to the answer",
#                     "items": {"type": "string"},
#                 },
#                 "options": {
#                     "type": "array",
#                     "description": "Four (4) additional wrong options for the question similar to the answer. Must be detailed and plausible, but wrong",
#                     "items": {"type": "string"},
#                 },
#             },
#             "required": ["question", "answer", "sentence_ids", "options"],
#         },
#     }
#     return prompt, function_dct
