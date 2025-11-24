#!/usr/bin/env python
"""Augment JSONL prompt/response datasets by generating paraphrases and adding metadata.

This is a lightweight, deterministic augmenter that applies simple templates
and synonym substitutions to create paraphrases. It's not a replacement for
human-curated examples, but it helps scale a small seed set into a larger
training corpus for PEFT experiments.

Usage:
    python scripts/augment_dataset.py --input data/general_search_dataset.jsonl \
        --output data/general_search_expanded.jsonl --multiplier 5

Output fields: `prompt`, `response`, `domain` (heuristic), `intent` (heuristic)

Note: For higher-quality paraphrases, consider using a paraphrasing model or human review.
"""

import argparse
import json
import random
import re
from pathlib import Path


SYNONYMS = {
    "how do i": ["how can I", "what's the best way to", "how should I"],
    "what is": ["what's", "explain", "tell me about"],
    "how to": ["how do you", "ways to", "steps to"],
    "best way": ["recommended way", "most effective method", "simplest way"],
    "remove": ["get rid of", "eliminate", "take out"],
}

TEMPLATES = [
    lambda p: p,
    lambda p: p + " Please provide concise steps.",
    lambda p: "Can you explain: " + p,
    lambda p: p.replace("How do I", "What's the best way to") if "How do I" in p else p,
    lambda p: "I want to know: " + p,
]


def heuristics_domain(prompt: str):
    p = prompt.lower()
    if re.search(r"(fever|doctor|prescribe|medic|dose|symptom|treatment)", p):
        return "medical"
    if re.search(r"(revenue|stock|investment|eps|market|q[0-9])", p):
        return "financial"
    if re.search(r"(pdf|word|excel|windows|mac|install|wifi|router)", p):
        return "tech"
    if re.search(r"(travel|hotel|flight|visa|airport)", p):
        return "travel"
    if re.search(r"(recipe|cook|rice|bake|kitchen)", p):
        return "cooking"
    return "general"


def heuristics_intent(prompt: str):
    p = prompt.lower()
    if p.strip().startswith("how") or p.strip().startswith("what") or p.strip().startswith("why"):
        return "informational"
    if p.strip().startswith("how do i") or p.strip().startswith("how to"):
        return "how-to"
    return "unknown"


def substitute_synonyms(text: str):
    t = text
    for k, vals in SYNONYMS.items():
        # case-insensitive replace
        pattern = re.compile(re.escape(k), flags=re.I)
        if pattern.search(t):
            rep = random.choice(vals)
            t = pattern.sub(rep, t, count=1)
    return t


def generate_paraphrases(prompt: str, n: int = 3):
    paraphrases = set()
    paraphrases.add(prompt)
    attempts = 0
    while len(paraphrases) < n and attempts < n * 4:
        p = prompt
        # apply synonym substitution sometimes
        if random.random() < 0.7:
            p = substitute_synonyms(p)
        # apply a random template
        tpl = random.choice(TEMPLATES)
        p = tpl(p)
        # minor punctuation/whitespace cleanups
        p = re.sub(r"\s+", " ", p).strip()
        paraphrases.add(p)
        attempts += 1
    return list(paraphrases)


def augment_file(input_path: Path, output_path: Path, multiplier: int = 5):
    out = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = obj.get("prompt") or obj.get("question") or ""
            response = obj.get("response") or obj.get("answer") or ""
            domain_hint = heuristics_domain(prompt)
            intent_hint = heuristics_intent(prompt)
            # generate paraphrases (including original)
            paras = generate_paraphrases(prompt, n=max(1, multiplier))
            for p in paras:
                out_obj = {
                    "prompt": p,
                    "response": response,
                    "domain": domain_hint,
                    "intent": intent_hint,
                }
                out.append(out_obj)

    # Optionally shuffle and write
    random.shuffle(out)
    with output_path.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multiplier", type=int, default=5, help="How many paraphrases per example (including original)")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    augment_file(inp, out, multiplier=args.multiplier)
    print(f"Wrote augmented dataset to {out} (multiplier={args.multiplier})")


if __name__ == "__main__":
    main()
