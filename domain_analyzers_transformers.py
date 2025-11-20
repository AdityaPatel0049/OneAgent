"""Transformer-backed domain analyzers using Hugging Face pipelines.

This module provides `analyze_financial`, `analyze_medical`, and
`analyze_generic` functions that attempt to use a zero-shot
classification model to detect domain signals. Each function falls
back gracefully to a simple TextBlob-based analysis if the
transformers libraries aren't available or initialization fails.

Note: `transformers` and a backend (torch or tensorflow) are required
to use the transformer analyzers. If not installed, the functions still
return a structured dict using TextBlob heuristics.
"""
from typing import Dict, List
import logging
import os
import json

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

from textblob import TextBlob
import requests

_CLASSIFIER = None


def _init_classifier():
    """Initialize a local HF zero-shot classifier pipeline if available.

    Returns pipeline or None.
    """
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return _CLASSIFIER
    if not _HAS_TRANSFORMERS:
        return None
    try:
        model = os.environ.get("HF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
        # device: -1 = cpu; if HF_USE_CUDA env var set to truthy, we attempt GPU
        device = 0 if os.environ.get("HF_USE_CUDA") in ("1", "true", "True") else -1
        _CLASSIFIER = pipeline("zero-shot-classification", model=model, device=device)
    except Exception as e:
        logging.exception("Failed to initialize transformer pipeline: %s", e)
        _CLASSIFIER = None
    return _CLASSIFIER


def _call_hf_inference_api(model_id: str, payload: Dict, token: str, timeout: int = 30) -> Dict:
    """Call the Hugging Face Inference API for a given `model_id`.

    - `model_id` should be the model slug (e.g. 'username/model-name').
    - `payload` is the JSON payload passed to the model. Typical simple usage is `{"inputs": text}`.
    - `token` is a HF Inference API token with `api:read` scope.

    Returns the JSON-decoded response or raises an exception on error.
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    # API typically returns JSON (list/dict) depending on model
    return resp.json()


def _fallback_analysis(text: str, domain_name: str) -> Dict:
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    score = round(max(0.0, min(1.0, 0.5 + 0.25 * (1.0 - subjectivity))), 3)
    issues: List[str] = []
    recs: List[str] = []
    highlights: List[str] = []
    if subjectivity > 0.6:
        issues.append("Text appears subjective")
        recs.append("Clarify opinion vs facts; cite sources for factual claims.")
    return {"domain": domain_name, "score": score, "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity}


def analyze_financial(text: str, weights: Dict[str, float] | None = None) -> Dict:
    """Transformer-backed financial analysis.

    Uses zero-shot labels to detect whether text contains:
    - actionable_advice, mentions_risk, contains_numbers, biased, objective
    The returned dict mirrors the structure used by the heuristic analyzers.
    """
    # Prefer calling a provided HF Inference API fine-tuned model if configured.
    hf_token = os.environ.get("HF_INFERENCE_API_TOKEN")
    hf_model = os.environ.get("HF_FINANCIAL_MODEL")
    if hf_token and hf_model:
        try:
            payload = {"inputs": text}
            resp = _call_hf_inference_api(hf_model, payload, hf_token)
            # Expect resp to be a list of {label, score} or a dict with useful keys.
            scores_by_label = {}
            if isinstance(resp, list):
                for obj in resp:
                    lab = obj.get("label")
                    sc = obj.get("score")
                    if lab:
                        scores_by_label[lab] = float(sc or 0.0)
            elif isinstance(resp, dict):
                # If the model returns a map of signals, try to use it directly
                # e.g. {"actionable":0.8, "risk":0.2}
                for k, v in resp.items():
                    try:
                        scores_by_label[str(k)] = float(v)
                    except Exception:
                        pass
            # map expected labels heuristically
            actionable = scores_by_label.get("actionable advice", scores_by_label.get("actionable", 0.0))
            risk = scores_by_label.get("mentions risk or volatility", scores_by_label.get("risk", 0.0))
            numeric = scores_by_label.get("contains numeric data", scores_by_label.get("numeric", 0.0))
            biased = scores_by_label.get("biased", 0.0)
            objective = scores_by_label.get("objective", 0.0)
            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
            if weights is None:
                weights = {}
            num_w = float(weights.get("numeric_weight", 0.45))
            subj_w = float(weights.get("subjectivity_weight", 0.35))
            act_w = float(weights.get("actionable_weight", 0.2))
            score = num_w * numeric + subj_w * (1.0 - subjectivity) + act_w * actionable
            score = max(0.0, min(1.0, score))
            issues: List[str] = []
            recs: List[str] = []
            highlights: List[str] = []
            if actionable > 0.5 and objective < 0.3:
                issues.append("Provides actionable advice with low objectivity")
                recs.append("Add data sources and risk disclosures for recommendations.")
            if risk > 0.4:
                highlights.append("Mentions risk or volatility")
            if numeric > 0.4:
                highlights.append("Contains numeric data")
            if biased > 0.6:
                issues.append("High likelihood of biased language")
            return {"domain": "financial", "score": round(score, 3), "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity, "signals": scores_by_label}
        except Exception as e:
            logging.exception("HF inference financial model call failed: %s", e)
            # fall through to local pipeline / heuristics

    # Otherwise fall back to zero-shot local pipeline (if installed) or simple heuristics
    clf = _init_classifier()
    if clf is None:
        return _fallback_analysis(text, "financial")

    labels = ["actionable advice", "mentions risk or volatility", "contains numeric data", "biased", "objective"]
    try:
        out = clf(text, candidate_labels=labels, hypothesis_template="This text {}.")
        scores_by_label = {lab: float(s) for lab, s in zip(out.get("labels", []), out.get("scores", []))}
    except Exception:
        return _fallback_analysis(text, "financial")

    actionable = scores_by_label.get("actionable advice", 0.0)
    risk = scores_by_label.get("mentions risk or volatility", 0.0)
    numeric = scores_by_label.get("contains numeric data", 0.0)
    biased = scores_by_label.get("biased", 0.0)
    objective = scores_by_label.get("objective", 0.0)

    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    if weights is None:
        weights = {}
    num_w = float(weights.get("numeric_weight", 0.45))
    subj_w = float(weights.get("subjectivity_weight", 0.35))
    act_w = float(weights.get("actionable_weight", 0.2))
    score = num_w * numeric + subj_w * (1.0 - subjectivity) + act_w * actionable
    score = max(0.0, min(1.0, score))

    issues: List[str] = []
    recs: List[str] = []
    highlights: List[str] = []

    if actionable > 0.5 and objective < 0.3:
        issues.append("Provides actionable advice with low objectivity")
        recs.append("Add data sources and risk disclosures for recommendations.")
    if risk > 0.4:
        highlights.append("Mentions risk or volatility")
    if numeric > 0.4:
        highlights.append("Contains numeric data")
    if biased > 0.6:
        issues.append("High likelihood of biased language")

    return {"domain": "financial", "score": round(score, 3), "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity, "signals": scores_by_label}


def analyze_medical(text: str, weights: Dict[str, float] | None = None) -> Dict:
    """Transformer-backed medical analysis.

    Uses zero-shot labels to detect clinical recommendations, urgency, and
    whether it advises consulting a professional.
    """
    # Prefer calling a provided HF Inference API fine-tuned model when configured.
    hf_token = os.environ.get("HF_INFERENCE_API_TOKEN")
    hf_model = os.environ.get("HF_MEDICAL_MODEL")
    if hf_token and hf_model:
        try:
            payload = {"inputs": text}
            resp = _call_hf_inference_api(hf_model, payload, hf_token)
            scores_by_label = {}
            if isinstance(resp, list):
                for obj in resp:
                    lab = obj.get("label")
                    sc = obj.get("score")
                    if lab:
                        scores_by_label[lab] = float(sc or 0.0)
            elif isinstance(resp, dict):
                for k, v in resp.items():
                    try:
                        scores_by_label[str(k)] = float(v)
                    except Exception:
                        pass

            clinical = scores_by_label.get("clinical recommendation or treatment", scores_by_label.get("clinical", 0.0))
            urgent = scores_by_label.get("urgent or emergency", scores_by_label.get("urgent", 0.0))
            consult = scores_by_label.get("recommends consulting a professional", scores_by_label.get("consult", 0.0))
            biased = scores_by_label.get("biased", 0.0)
            objective = scores_by_label.get("objective", 0.0)

            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)

            if weights is None:
                weights = {}
            consult_w = float(weights.get("consult_weight", 0.4))
            subj_w = float(weights.get("subjectivity_weight", 0.3))
            obj_w = float(weights.get("objective_weight", 0.3))
            score = consult_w * consult + subj_w * (1.0 - subjectivity) + obj_w * objective
            score = max(0.0, min(1.0, score))

            issues: List[str] = []
            recs: List[str] = []
            highlights: List[str] = []

            if clinical > 0.5 and consult < 0.4:
                issues.append("Gives clinical recommendations without urging consultation")
                recs.append("Add a clear suggestion to consult a qualified healthcare professional and cite guidelines.")
            if urgent > 0.4:
                issues.append("Uses urgent/emergency language — ensure triage guidance is clear")
            if consult > 0.5:
                highlights.append("Suggests consulting a professional")

            return {"domain": "medical", "score": round(score, 3), "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity, "signals": scores_by_label}
        except Exception as e:
            logging.exception("HF inference medical model call failed: %s", e)
            # fall through to local pipeline / heuristics

    # Otherwise use local zero-shot pipeline or fallback heuristics
    clf = _init_classifier()
    if clf is None:
        return _fallback_analysis(text, "medical")

    labels = ["clinical recommendation or treatment", "urgent or emergency", "recommends consulting a professional", "biased", "objective"]
    try:
        out = clf(text, candidate_labels=labels, hypothesis_template="This text {}.")
        scores_by_label = {lab: float(s) for lab, s in zip(out.get("labels", []), out.get("scores", []))}
    except Exception:
        return _fallback_analysis(text, "medical")

    clinical = scores_by_label.get("clinical recommendation or treatment", 0.0)
    urgent = scores_by_label.get("urgent or emergency", 0.0)
    consult = scores_by_label.get("recommends consulting a professional", 0.0)
    biased = scores_by_label.get("biased", 0.0)
    objective = scores_by_label.get("objective", 0.0)

    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    if weights is None:
        weights = {}
    consult_w = float(weights.get("consult_weight", 0.4))
    subj_w = float(weights.get("subjectivity_weight", 0.3))
    obj_w = float(weights.get("objective_weight", 0.3))
    score = consult_w * consult + subj_w * (1.0 - subjectivity) + obj_w * objective
    score = max(0.0, min(1.0, score))

    issues: List[str] = []
    recs: List[str] = []
    highlights: List[str] = []

    if clinical > 0.5 and consult < 0.4:
        issues.append("Gives clinical recommendations without urging consultation")
        recs.append("Add a clear suggestion to consult a qualified healthcare professional and cite guidelines.")
    if urgent > 0.4:
        issues.append("Uses urgent/emergency language — ensure triage guidance is clear")

    if consult > 0.5:
        highlights.append("Suggests consulting a professional")

    return {"domain": "medical", "score": round(score, 3), "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity, "signals": scores_by_label}


def analyze_generic(text: str, weights: Dict[str, float] | None = None) -> Dict:
    # Prefer calling a provided HF Inference API fine-tuned model when configured.
    hf_token = os.environ.get("HF_INFERENCE_API_TOKEN")
    hf_model = os.environ.get("HF_GENERIC_MODEL")
    if hf_token and hf_model:
        try:
            payload = {"inputs": text}
            resp = _call_hf_inference_api(hf_model, payload, hf_token)
            scores_by_label = {}
            if isinstance(resp, list):
                for obj in resp:
                    lab = obj.get("label")
                    sc = obj.get("score")
                    if lab:
                        scores_by_label[lab] = float(sc or 0.0)
            elif isinstance(resp, dict):
                for k, v in resp.items():
                    try:
                        scores_by_label[str(k)] = float(v)
                    except Exception:
                        pass

            biased = scores_by_label.get("biased", 0.0)
            objective = scores_by_label.get("objective", 0.0)
            numeric = scores_by_label.get("contains numeric data", scores_by_label.get("numeric", 0.0))

            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)

            if weights is None:
                weights = {}
            subj_w = float(weights.get("subjectivity_weight", 0.4))
            obj_w = float(weights.get("objective_weight", 0.4))
            num_w = float(weights.get("numeric_weight", 0.2))
            score = subj_w * (1.0 - subjectivity) + obj_w * objective + num_w * numeric
            score = max(0.0, min(1.0, score))

            issues: List[str] = []
            recs: List[str] = []
            highlights: List[str] = []
            if biased > 0.6:
                issues.append("High likelihood of biased language")
            if numeric > 0.4:
                highlights.append("Contains numeric data")

            return {"domain": "generic", "score": round(score, 3), "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity, "signals": scores_by_label}
        except Exception as e:
            logging.exception("HF inference generic model call failed: %s", e)
            # fall through to local pipeline / heuristics

    # Otherwise fall back to local zero-shot classifier or heuristics
    clf = _init_classifier()
    if clf is None:
        return _fallback_analysis(text, "generic")

    labels = ["biased", "objective", "contains numeric data"]
    try:
        out = clf(text, candidate_labels=labels, hypothesis_template="This text {}.")
        scores_by_label = {lab: float(s) for lab, s in zip(out.get("labels", []), out.get("scores", []))}
    except Exception:
        return _fallback_analysis(text, "generic")

    biased = scores_by_label.get("biased", 0.0)
    objective = scores_by_label.get("objective", 0.0)
    numeric = scores_by_label.get("contains numeric data", 0.0)

    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    if weights is None:
        weights = {}
    subj_w = float(weights.get("subjectivity_weight", 0.4))
    obj_w = float(weights.get("objective_weight", 0.4))
    num_w = float(weights.get("numeric_weight", 0.2))
    score = subj_w * (1.0 - subjectivity) + obj_w * objective + num_w * numeric
    score = max(0.0, min(1.0, score))

    issues: List[str] = []
    recs: List[str] = []
    highlights: List[str] = []
    if biased > 0.6:
        issues.append("High likelihood of biased language")
    if numeric > 0.4:
        highlights.append("Contains numeric data")

    return {"domain": "generic", "score": round(score, 3), "issues": issues, "recommendations": recs, "highlights": highlights, "polarity": polarity, "subjectivity": subjectivity, "signals": scores_by_label}
