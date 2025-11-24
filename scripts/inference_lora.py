#!/usr/bin/env python
"""Load a base model and PEFT (LoRA) adapters, run prompts, and print generations.

Usage examples:
  # single prompt
  python scripts\inference_lora.py --base_model gpt2 --peft_path lora-output --prompt "How do I cook rice?"

  # prompts file (one prompt per line)
  python scripts\inference_lora.py --base_model gpt2 --peft_path lora-output --prompts_file prompts.txt

Options include `--load_in_8bit` for bitsandbytes-backed models.
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="Base model name or path (e.g., gpt2)")
    p.add_argument("--peft_path", required=True, help="Path to PEFT/LoRA adapter folder (output of training)")
    p.add_argument("--prompt", default=None, help="Single prompt to run")
    p.add_argument("--prompts_file", default=None, help="File with prompts (one per line)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--num_return_sequences", type=int, default=1)
    p.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit (requires bitsandbytes)")
    return p.parse_args()


def load_model(base_model, peft_path, load_in_8bit=False):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (optionally in 8-bit)
    kwargs = {}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    # Wrap with PEFT adapters
    model = PeftModel.from_pretrained(model, peft_path)

    model.eval()
    return tokenizer, model


def run_generation(tokenizer, model, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    # move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
    return texts


def main():
    args = parse_args()

    if args.prompt is None and args.prompts_file is None:
        print("Provide --prompt or --prompts_file", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = load_model(args.base_model, args.peft_path, load_in_8bit=args.load_in_8bit)

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)

    for i, p in enumerate(prompts, 1):
        print(f"\n=== Prompt {i} ===")
        print(p)
        gens = run_generation(
            tokenizer,
            model,
            p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
        )
        for j, g in enumerate(gens, 1):
            # strip the prompt portion from generation for clarity when possible
            if g.startswith(p):
                out_text = g[len(p):].strip()
            else:
                out_text = g
            print(f"\n--- Generation {j} ---\n{out_text}\n")


if __name__ == "__main__":
    main()
