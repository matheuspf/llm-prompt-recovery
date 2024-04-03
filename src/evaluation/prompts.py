import re
import json

def get_prompt_subject(dct):
    prompt = f'''[INST] Given an original text and the rewritten version of it from a LLM, predict what was the subject of the used prompt for the rewrite.

Original text:
""""""
{dct["original_text"]}
""""""

Rewritten text:
""""""
{dct["rewritten_text"]}
""""""

Subject:
""""""
[/INST] 
'''
    return prompt



def get_prompt_subject_quality(dct):
    prompt = f'''[INST] Given an original text and the rewritten version of it from a LLM, predict what was the subject of the used prompt for the rewrite as well as if the rewrite was successful or not.

Original text:
""""""
{dct["original_text"]}
""""""

Rewritten text:
""""""
{dct["rewritten_text"]}
""""""

Subject: [/INST] '''
    return prompt



def get_prompt_prompt(dct):
    prompt = f'''[INST] Given an original text and the rewritten version of it from a LLM, predict what was the Prompt used for the rewrite.

Original text:
""""""
{dct["original_text"]}
""""""

Rewritten text:
""""""
{dct["rewritten_text"]}
""""""

Prompt:
""""""
[/INST] 
'''
    return prompt


def get_merge_prompt_subject(subject):
    prompt = \
f'''[INST] Your task is to rewrite a prompt in order to introduce a subject in it.

You will be given:

- Prompt: the prompt that is going to be rewritten using the subject
- Subject: the subject to be introduced into the prompt
- Output: Your output, rewritting `Prompt` to have subject `Subject`

Pay attention to the given examples and try to replicate exactly the same text style.

Examples:

Prompt: Improve the text to this.

-----------------------------------------------------------
[/INST] 
Subject: chronological timeline
Output: Improve the text to be a chronological timeline.

Subject: cultural references, specific region
Output: Improve the text to include cultural references from a specific region.

Subject: onomatopoeia, vivid description
Output: Improve the text to sound like onomatopoeia, with vivid descriptions.

Subject: child's perspective
Output: Improve the text to be from a child's perspective.

Subject: academic language, research-oriented
Output: Improve the text to a more academic, research-oriented language.

Subject: mysterious, cryptic clues
Output: Improve the text to include mysterious, cryptic clues.

Subject: cooking recipe
Output: Improve the text into a cooking recipe.

Subject: horror elements
Output: Improve the text to infuse horror elements.

Subject: suspenseful, cliffhangers
Output: Improve the text to be suspenseful, with cliffhangers.

Subject: whimsical, fantastical tone
Output: Improve the text to have a whimsical, fantastical tone.

Subject: metaphysical concepts, philosophical depth
Output: Improve the text to include metaphysical concepts with philosophical depth.

Subject: pet's perspective
Output: Improve the text to be from a pet's perspective.

Subject: tweets
Output: Improve the text to be in a tweet format.

Subject: weather forecast report
Output: Improve the text sound like a weather forecast report.

Subject: rhyming poem
Output: Improve the text to be rhyme like a poem.

Subject: {subject}
Output: '''
    return prompt



def get_merge_prompt_subject_2(subject):
    prompt = \
f'''[INST] Your task is to rewrite a prompt in order to introduce a subject in it.

You will be given:

- Prompt: the prompt that is going to be rewritten using the subject
- Subject: the subject to be introduced into the prompt
- Output: Your output, rewritting `Prompt` to have subject `Subject`

Pay attention to the given examples and try to replicate exactly the same text style.

Examples:

Prompt: Please improve the following text rewriting it as a ...

-----------------------------------------------------------
[/INST] 
Subject: chronological timeline
Output: Please improve the following text rewriting it in a chronological timeline.

Subject: cultural references, specific region
Output: Please improve the following text rewriting it with cultural references from a specific region.

Subject: onomatopoeia, vivid description
Output: Please improve the following text rewriting it with onomatopoeia and vivid descriptions.

Subject: child's perspective
Output: Please improve the following text rewriting it from a child's perspective.

Subject: academic language, research-oriented
Output: Please improve the following text rewriting it with a more academic, research-oriented language.

Subject: mysterious, cryptic clues
Output: Please improve the following text rewriting it with mysterious, cryptic clues.

Subject: cooking recipe
Output: Please improve the following text rewriting it as a cooking recipe.

Subject: horror elements
Output: Please improve the following text rewriting it with horror elements.

Subject: suspenseful, cliffhangers
Output: Please improve the following text rewriting it to be suspenseful, with cliffhangers.

Subject: whimsical, fantastical tone
Output: Please improve the following text rewriting it to have a whimsical, fantastical tone.

Subject: metaphysical concepts, philosophical depth
Output: Please improve the following text rewriting it with metaphysical concepts and philosophical depth.

Subject: pet's perspective
Output: Please improve the following text rewriting it from a pet's perspective.

Subject: tweets
Output: Please improve the following text rewriting it in a tweet format.

Subject: weather forecast report
Output: Please improve the following text rewriting it to sound like a weather forecast report.

Subject: rhyming poem
Output: Please improve the following text rewriting it to rhyme like a poem.

Subject: {subject}
Output: '''
    return prompt



def get_prompt_json(dct):
    json_input = {
        "original_text": dct["original_text"],
        "rewritten_text": dct["rewritten_text"]
    }
    
    return f"""<s> [INST]
I will give you a JSON with following structure:
{{
    'original_text': 'An original piece of text.'
    'rewritten_text': 'A version of original_text that was rewritten by an LLM according to a specific prompt.'
}}

Given the task of understanding how text is rewritten by analyzing the original_text and rewritten_text, your goal is to deduce the specific instructions or prompt that was most likely used to generate the rewritten text from the original text. Consider the changes made in terms of style, tone, structure, and content. Assess whether the rewrite focuses on summarization, paraphrasing, stylistic alteration (e.g., formal to informal), or any specific content changes (e.g., making the text more concise, expanding on ideas, or altering the perspective). Follow this steps:

1. Read the original_text: Start by thoroughly understanding the content, style, tone, and purpose of the original text. Note any key themes, technical terms, and the overall message.
2. Analyze the rewritten_text: Examine how the rewritten text compares to the original. Identify what has been changed, added, or omitted. Pay close attention to changes in style (formal, informal), tone (serious, humorous), structure (paragraph order, sentence structure), and any shifts in perspective or emphasis.
3. Infer the Prompt subject: Based on your analysis, infer the most likely prompt content that guided the rewriting process. Your inference should account for the observed changes in style, tone, structure, and content.

Based on your analysis return the subject of the prompt, a few words containing the main idea of it.

Return your answer using the following JSON structure:
{{"subject": "Your best guess for the prompt subject used to generate the rewritten text from the original text."}}

Return a valid JSON as output and nothing more.

-----------------------
Input: {json.dumps(json_input)} [/INST] """


def json_parser_from_chat_response(text: str) -> dict:
    try:
        # Greedy search for 1st json candidate.
        match = re.search(
            r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        json_str = ""
        if match:
            json_str = match.group()
        json_object = json.loads(json_str, strict=False)
        return json_object
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse json from completion {text}. Got: {e}")

