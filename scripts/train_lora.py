#!/usr/bin/env python
"""Simple LoRA/PEFT training script using Hugging Face Transformers + PEFT.

This is a minimal example intended to run on a machine with a GPU.
It expects a JSONL dataset with fields `prompt` and `response` (see
`data/example_dataset.jsonl`). It uses Trainer for convenience and
PEFT/LoRA for parameter-efficient fine-tuning.

Notes:
- This script assumes you have `bitsandbytes`, `accelerate`, `transformers`,
  `datasets`, and `peft` installed. See updated `requirements.txt`.
- For memory-limited GPUs, use `--load_in_8bit` to enable 8-bit loading (bitsandbytes).
"""

import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2", help="Base model name or path")
    p.add_argument("--data", default="data/example_dataset.jsonl", help="Path to JSONL dataset")
    p.add_argument("--output_dir", default="lora-output", help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    return p.parse_args()


def make_prompt(example):
    # Combine prompt+response into a single text for causal LM training.
    # Keep the delimiter stable so model learns where response begins.
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    return prompt + "\n\n### Response:\n" + response


def main():
    args = parse_args()

    ds = load_dataset("json", data_files=args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        texts = [make_prompt(x) for x in examples["data"]] if "data" in examples else [make_prompt(x) for x in examples]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_length)

    # dataset may have 'train' key; if not, treat the loaded dataset itself as train
    if "train" in ds:
        train_ds = ds["train"].map(lambda ex: tokenize_fn(ex), batched=True, remove_columns=ds["train"].column_names)
    else:
        # when calling load_dataset with a local json file, the dataset is usually a single split
        split = list(ds.keys())[0]
        train_ds = ds[split].map(lambda ex: tokenize_fn(ex), batched=True, remove_columns=ds[split].column_names)

    # prepare model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=args.load_in_8bit,
        device_map="auto" if args.load_in_8bit else None,
    )

    # If using k-bit (8-bit) we should prepare the model for k-bit training
    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=not args.load_in_8bit,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
