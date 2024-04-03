# import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from typing import Dict, List, Optional, Union
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names



import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    GPTQConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import DataCollatorForLanguageModeling
from unsloth import FastLanguageModel


def get_training_args():
    return TrainingArguments(
        output_dir="./results",
        max_steps=3000,
        # num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        # optim="paged_adamw_32bit",
        optim="adamw_8bit",
        save_steps=200,
        eval_steps=200,
        logging_steps=20,
        evaluation_strategy="steps",
        learning_rate=1e-4,
        weight_decay=1e-5,
        fp16=False,
        # tf32=True,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        # group_by_length=True,
        lr_scheduler_type="constant",
        # report_to="wandb",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=50
    )


def get_peft_config():
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.2,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )


def get_model(model_id: str):
    # quant_config = GPTQConfig(bits=4, disable_exllama=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        # quantization_config=quant_config,
        load_in_4bit=True,
        max_seq_length=2048,
        # torch_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # model.gradient_checkpointing_enable()

    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_dropout=0., # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=27001,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=False, # And LoftQ
    )

    return model, tokenizer


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["original_text"])):
        original_text = example["original_text"][i]
        rewritten_text = example["rewritten_text"][i]
        prompt = example["rewrite_prompt"][i]
        subject = example["subject"][i]
        # success = "Yes" if bool(example["is_well_written"][i]) else "No"

        text = f'''[INST] Given an original text and the rewritten version of it from a LLM, predict what was the prompt and subject used for the rewrite.

Original text:
""""""
{original_text}
""""""

Rewritten text:
""""""
{rewritten_text}
""""""
[/INST]

Prompt: {prompt}
Subject: {subject}
'''

        output_texts.append(text)
    return output_texts


def get_data_collator(tokenizer):
    user_tag = '"\n[/INST]\n'
    response_template_ids = tokenizer.encode(user_tag, add_special_tokens=False)[2:-1]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    return data_collator


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)



def train(model_id: str, dataset_id: str):
    model, tokenizer = get_model(model_id)

    # peft_config = get_peft_config()
    # model = prepare_model_for_kbit_training(
    #     model,
    #     use_gradient_checkpointing=True,
    #     gradient_checkpointing_kwargs={"use_reentrant": False},
    # )
    # model = get_peft_model(model, peft_config)

    dataset = DatasetDict.load_from_disk(dataset_id)

    training_arguments = get_training_args()
    data_collator = get_data_collator(tokenizer)
    
    # adam_bnb_optim = get_optimizer(model, training_arguments)

    # trainer = CustomTrainer(
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=T5CosSimMetric(),
        packing=False,
        # optimizers=(adam_bnb_optim, None)
        dataset_num_proc=4
    )

    trainer.train()
    trainer.model.save_pretrained("./results/finetuned_mistral")
    trainer.tokenizer.save_pretrained("./results/finetuned_mistral")


if __name__ == "__main__":
    # train("mistralai/Mistral-7B-Instruct-v0.2", "/kaggle/input/llm-prompt-rewriting-dataset")
    train("unsloth/mistral-7b-instruct-v0.2-bnb-4bit", "/kaggle/input/llm-prompt-rewriting-dataset")
    # train("teknium/OpenHermes-2.5-Mistral-7B", "/kaggle/input/llm-prompt-rewriting-dataset")
    # train("Open-Orca/OpenOrca-Platypus2-13B", "/kaggle/input/llm-prompt-rewriting-dataset")
