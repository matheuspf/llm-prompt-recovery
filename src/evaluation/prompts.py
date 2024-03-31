
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



