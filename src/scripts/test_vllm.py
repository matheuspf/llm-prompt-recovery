from vllm import LLM, SamplingParams

# llm = LLM("TheBloke/Yi-34B-Chat-AWQ", quantization="awq", dtype="half", trust_remote_code=True)
llm = LLM("TheBloke/Yi-34B-Chat-GPTQ", quantization="gptq", dtype="half", trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.5, top_p=0.95, top_k=-1, max_tokens=128)

prompt = "Tell me a data science joke"
prompt = f"""Human: {prompt} Assistant:
"""

output = llm.generate([prompt], sampling_params)[0]

prompt = output.prompt
generated_text = output.outputs[0].text
print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
