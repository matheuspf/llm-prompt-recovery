import torch
from tqdm import tqdm
from typing import Optional, Union, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, GenerationConfig, GPTQConfig
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import DatasetDict, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def get_training_args():
    return TrainingArguments(
        output_dir="./results",
        max_steps=5000,
        # num_train_epochs=10,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        # optim="adamw_8bit",
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        evaluation_strategy="steps",
        metric_for_best_model="eval_loss",
        learning_rate=1e-4,
        weight_decay=1e-5,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        # report_to="wandb"
    )


def get_peft_config():
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )


def get_model(model_id: str):
    # quant_config = GPTQConfig(bits=4, disable_exllama=True)
    quant_config = BitsAndBytesConfig(  
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False 
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    return model, tokenizer




def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['original_text'])):
        original_text = example['original_text'][i]
        rewritten_text = example['rewritten_text'][i]
        prompt = example['subject'][i]
        
        

        text = \
f'''[INST] Given an original text and the rewritten version of it from a LLM, predict what was the subject of the used prompt for the rewrite.

Original text:
""""""
{original_text}
""""""

Rewritten text:
""""""
{rewritten_text}
""""""

Subject:
""""""
[/INST] {prompt}
'''
        output_texts.append(text)
    return output_texts



# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['original_text'])):
#         original_text = example['original_text'][i]
#         rewritten_text = example['rewritten_text'][i]
#         prompt = example['rewrite_prompt'][i]

#         text = \
# f'''[INST] Given the task of understanding how text is rewritten by analyzing the Original Text and Rewritten Text, your goal is to deduce the specific instructions or prompt that was most likely used to generate the rewritten text from the original text. Consider the changes made in terms of style, tone, structure, and content. Assess whether the rewrite focuses on summarization, paraphrasing, stylistic alteration (e.g., formal to informal), or any specific content changes (e.g., making the text more concise, expanding on ideas, or altering the perspective). Follow this steps:

# 1. Read the Original Text: Start by thoroughly understanding the content, style, tone, and purpose of the original text. Note any key themes, technical terms, and the overall message.
# 2. Analyze the Rewritten Text: Examine how the rewritten text compares to the original. Identify what has been changed, added, or omitted. Pay close attention to changes in style (formal, informal), tone (serious, humorous), structure (paragraph order, sentence structure), and any shifts in perspective or emphasis.
# 3. Infer the Prompt: Based on your analysis, infer the most likely prompt that guided the rewriting process. Your inference should account for the observed changes in style, tone, structure, and content. Specify the type of task (e.g., summarize, paraphrase, make more accessible to a general audience), any specific directions evident from the changes, and any specific stylistic choice (e.g., 'in the S')

# Based on your analysis return the prompt as if you were given the instruction your self like:
# "Rewrite this text..."
# "Transform this ... into ... based on the style of ..."

# Make the prompt short and direct using a maximum of 20 words.


# Original Text
# """"""
# {original_text}
# """"""

# Rewritten text:
# """"""
# {rewritten_text}
# """"""

# Prompt:
# """"""
# [/INST] {prompt}
# '''
#         output_texts.append(text)
#     return output_texts



def get_data_collator(tokenizer):
    user_tag = '"\n[/INST] '
    response_template_ids = tokenizer.encode(user_tag, add_special_tokens=False)[2:-1]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    return data_collator


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


class T5CosSimMetric:
    def __init__(self):
        self.t5 = SentenceTransformer("sentence-transformers/sentence-t5-base")
        self.gts_embds = None

    def __call__(self, gts_text, preds_text):
        if self.gts_embds is None:
            self.gts_embds = self.t5.encode(gts_text, normalize_embeddings=True, batch_size=32)
        
        gts_embds = self.gts_embds
        preds_embds = self.t5.encode(preds_text, normalize_embeddings=True, batch_size=32)
        cos_sim = torch.nn.functional.cosine_similarity(gts_embds, preds_embds, dim=1)
        cos_sim = (cos_sim ** 3)

        metric = cos_sim.mean().item()

        return metric


class CustomTrainer(SFTTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset or self.eval_dataset
        data = self.data_collator(eval_dataset["input_ids"])
        data["input_ids"] = data["input_ids"].to(self.args.device)
        
        preds_text = []
        gts_text = []
        
        pbar = tqdm(zip(data["input_ids"], data["labels"]), total=len(data["input_ids"]))
        for input_ids, labels in pbar:
            with torch.no_grad():
                pad_pos = (input_ids == self.tokenizer.pad_token_id).long().argmax()
                labels_pos = (labels != -100).long().argmax()
                
                labels = labels[labels_pos:pad_pos]
                input_ids = input_ids[:pad_pos - len(labels)]

                pred_ids = self.model.generate(
                    input_ids.unsqueeze(0),
                    generation_config=GenerationConfig(
                        use_cache=False,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    ))[0]

                gt_text = self.tokenizer.decode(labels)
                pred_text = self.tokenizer.decode(pred_ids[input_ids.shape[0]:])
                
                gts_text.append(gt_text)
                preds_text.append(pred_text)
        
        t5_cos_sim = self.compute_metrics(gts_text, preds_text)
        metrics = {f"{metric_key_prefix}_t5_cos_sim": t5_cos_sim}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics



def train(model_id: str, dataset_id: str):
    model, tokenizer = get_model(model_id)

    peft_config = get_peft_config()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)

    # dataset = load_dataset(dataset_id)
    dataset = DatasetDict.load_from_disk(dataset_id)

    training_arguments = get_training_args()
    data_collator = get_data_collator(tokenizer)
    
    # trainer = CustomTrainer(
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=T5CosSimMetric(),
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained("./results/finetuned_mistral")


if __name__ == "__main__":
    train(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "/kaggle/input/llm-prompt-rewriting-dataset"
    )
