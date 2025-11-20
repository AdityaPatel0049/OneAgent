"""Domain-specific analyzers for agent responses.

These are lightweight, rule- and heuristic-based analyzers intended
to augment the existing sentiment/objectivity pipeline. Each analyzer
returns a dict with a domain score, issues, and short recommendations.

The functions are intentionally simple so they run without heavy ML
dependencies; they can later be replaced by calls to fine-tuned
transformers or external APIs.
"""
from typing import Dict, List
import re
from textblob import TextBlob


def _count_numbers(text: str) -> int:
    return len(re.findall(r"\d+(?:[\.,]\d+)?", text))


def analyze_financial(text: str, weights: Dict[str, float] | None = None) -> Dict:
    """Return a lightweight financial analysis for `text`.

    `weights` can be used to override scoring weights. Supported keys:
      - base: base score (default 0.5)
      - numeric_weight: weight for numeric evidence (default 0.15)
      - numeric_divisor: divisor for numeric normalization (default 5.0)
      - objectivity_weight: weight for objectivity (default 0.2)
      - actionable_penalty: penalty when explicit recommendations appear (default 0.05)
    """
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    issues: List[str] = []
    recommendations: List[str] = []
    highlights: List[str] = []

    # Detect if there are explicit investment recommendations
    rec_keywords = [r"\bbuy\b", r"\bsell\b", r"\bhold\b", r"\brecommend\b", r"\bshould invest\b"]
    if any(re.search(k, text, flags=re.I) for k in rec_keywords):
        highlights.append("Contains explicit investment language")

    # Risk language and disclaimers
    if re.search(r"\brisk\b|\brisk[s]?\b|loss|volatile|uncertain", text, flags=re.I):
        highlights.append("Mentions risk or volatility")
    else:
        issues.append("No explicit mention of risks or uncertainty — consider adding risk disclosure")

    # Check for numeric evidence
    num_count = _count_numbers(text)
    if num_count >= 2:
        highlights.append(f"Includes {num_count} numeric data points")
    else:
        issues.append("Few or no numeric data points cited; consider adding figures or citations")

    # Avoid strong prescriptive language without references
    if subjectivity > 0.5 and abs(polarity) > 0.3:
        issues.append("Text is opinionated; avoid prescriptive investment advice without citations")

    # Basic confidence scoring: combine subject-matter signals
    if weights is None:
        weights = {}
    base = float(weights.get("base", 0.5))
    numeric_weight = float(weights.get("numeric_weight", 0.15))
    numeric_divisor = float(weights.get("numeric_divisor", 5.0))
    objectivity_weight = float(weights.get("objectivity_weight", 0.2))
    actionable_penalty = float(weights.get("actionable_penalty", 0.05))

    score = base
    score += numeric_weight * min(1.0, num_count / numeric_divisor)
    score += objectivity_weight * (1.0 - subjectivity)
    if any(re.search(k, text, flags=re.I) for k in rec_keywords):
        score -= actionable_penalty

    # clamp
    score = max(0.0, min(1.0, score))

    if "No explicit mention of risks" in " ".join(issues):
        recommendations.append("Add risk considerations and a brief disclaimer that this is not financial advice.")
    if num_count < 2:
        recommendations.append("Cite figures, percentages, or sources to back quantitative claims.")
    if subjectivity > 0.6:
        recommendations.append("Reduce subjective wording or label opinions clearly.")

    return {
        "domain": "financial",
        "score": round(score, 3),
        "issues": issues,
        "recommendations": recommendations,
        "highlights": highlights,
        "polarity": polarity,
        "subjectivity": subjectivity,
    }


def analyze_medical(text: str, weights: Dict[str, float] | None = None) -> Dict:
    """Return a lightweight medical analysis for `text`.

    Output is similar to `analyze_financial` but focused on safety, advice,
    and presence of clinical guidance vs. high-level information.
    """
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    issues: List[str] = []
    recommendations: List[str] = []
    highlights: List[str] = []

    # Detect whether it includes clinical recommendations
    clinical_keywords = [r"\bdiagnos(e|is)\b", r"\btreatment\b", r"\bprescribe\b", r"\bmedicat(e|ion)\b", r"\badvice\b"]
    if any(re.search(k, text, flags=re.I) for k in clinical_keywords):
        highlights.append("Contains clinical or treatment-related language")

    # Check for safety disclaimers / consult language
    if re.search(r"consult (a|your) (doctor|physician|healthcare|provider)|seek medical attention|see a doctor", text, flags=re.I):
        highlights.append("Includes 'consult a professional' guidance")
    else:
        issues.append("No suggestion to consult a qualified healthcare professional when appropriate")

    # Alarm words that should trigger caution
    if re.search(r"\bimmediate|urgent|emergency|life[- ]threaten", text, flags=re.I):
        issues.append("Uses urgent/emergency language — ensure clear triage advice is present")

    # Evidence/citation heuristic (numbers and references)
    num_count = _count_numbers(text)
    if num_count >= 1:
        highlights.append(f"Includes {num_count} numeric data points (may be study results or dosages)")
    else:
        recommendations.append("Reference clinical studies or authoritative sources when asserting clinical claims")

    # Confidence scoring: penalize subjective clinical advice
    if weights is None:
        weights = {}
    base = float(weights.get("base", 0.6))
    numeric_weight = float(weights.get("numeric_weight", 0.1))
    numeric_divisor = float(weights.get("numeric_divisor", 3.0))
    objectivity_weight = float(weights.get("objectivity_weight", 0.2))
    clinical_penalty = float(weights.get("clinical_penalty", 0.15))

    score = base
    score += numeric_weight * min(1.0, num_count / numeric_divisor)
    score += objectivity_weight * (1.0 - subjectivity)
    if any(re.search(k, text, flags=re.I) for k in clinical_keywords) and subjectivity > 0.6:
        score -= clinical_penalty

    score = max(0.0, min(1.0, score))

    if re.search(r"\bnever\b|\ballow\b", text, flags=re.I) and not re.search(r"(consult|evidence|study|guideline)", text, flags=re.I):
        issues.append("Absolute language detected without cited evidence")
        recommendations.append("Avoid absolute statements about treatments without citing guidelines or studies")

    return {
        "domain": "medical",
        "score": round(score, 3),
        "issues": issues,
        "recommendations": recommendations,
        "highlights": highlights,
        "polarity": polarity,
        "subjectivity": subjectivity,
    }


def analyze_generic(text: str, weights: Dict[str, float] | None = None) -> Dict:
    """Fallback generic analysis — returns basic polarity/subjectivity and simple tips."""
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    if weights is None:
        weights = {}
    base = float(weights.get("base", 0.5))
    subjectivity_weight = float(weights.get("subjectivity_weight", 0.25))
    score = round(max(0.0, min(1.0, base + subjectivity_weight * (1.0 - subjectivity))), 3)

    issues = []
    recommendations = []
    if subjectivity > 0.6:
        issues.append("Text appears subjective")
        recommendations.append("Clarify which statements are opinion vs. facts, and cite sources for factual claims.")

    return {
        "domain": "generic",
        "score": score,
        "issues": issues,
        "recommendations": recommendations,
        "highlights": [],
        "polarity": polarity,
        "subjectivity": subjectivity,
    }
