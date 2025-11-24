import re
from textblob import TextBlob
from typing import Dict


def _count_numbers(text: str) -> int:
    return len(re.findall(r"\d+(?:[\.,]\d+)?", text or ""))


def auto_weights(text: str, agent: str | None = None, use_transformers: bool = False) -> Dict[str, Dict]:
    """Compute automatic analyzer weight dictionaries based on text signals and agent.

    Heuristics used (simple, interpretable rules):
      - Count numeric evidence: more numbers -> increase numeric weight
      - Subjectivity: more subjective text -> increase objectivity weight impact
      - Agent type: financial/medical prefer higher base and numeric weight
      - Presence of domain keywords (recommend/buy/sell for finance, diagnos/treatment for med)

    Returns a dict with keys: `weights_fin`, `weights_med`, `weights_gen`,
    and transformer variants `tf_weights_fin`, `tf_weights_med`, `tf_weights_gen`.

    These rules are intentionally conservative and transparent; tune as needed.
    """
    if text is None:
        text = ""
    blob = TextBlob(text)
    subjectivity = float(blob.sentiment.subjectivity)
    polarity = float(blob.sentiment.polarity)
    # Raw numeric counts and type-aware counts
    num_count = _count_numbers(text)
    pct_count = len(re.findall(r"\b\d+(?:[\.,]\d+)?%\b", text or ""))
    currency_count = len(re.findall(r"\b(?:\$|€|£|₹)\s?\d+(?:[\.,]\d+)?\b", text or ""))
    # effective_count: weight different numeric evidence types
    # percentages and currency are usually stronger signals
    effective_count = float(num_count) + 0.5 * float(pct_count) + 0.8 * float(currency_count)
    length = len(text or "")

    # presence flags
    fin_recs = bool(re.search(r"\b(buy|sell|hold|recommend|should invest|investment)\b", text, flags=re.I))
    med_clinical = bool(re.search(r"\b(diagnos(e|is)|treatment|prescribe|medicat|medication|dose|dosage)\b", text, flags=re.I))

    # Base values
    base_fin = 0.5
    base_med = 0.6
    base_gen = 0.5

    # Adjust base by agent if specified and apply a competitive bias so
    # the selected agent is boosted while other domains are attenuated.
    selected = None
    if agent:
        al = agent.lower()
        if "financial" in al:
            selected = "financial"
            base_fin = min(0.95, base_fin + 0.2)
        if "medical" in al:
            selected = "medical"
            base_med = min(0.95, base_med + 0.2)
        # Treat generic/general selection as choosing the generic expert
        if "general" in al or "generic" in al:
            selected = "generic"
            base_gen = min(0.95, base_gen + 0.2)

    # Attenuate non-selected domains so the chosen expert becomes more competitive.
    if selected == "financial":
        base_med = base_med * 0.6
        base_gen = base_gen * 0.7
    elif selected == "medical":
        base_fin = base_fin * 0.6
        base_gen = base_gen * 0.7
    elif selected == "generic":
        # When the user explicitly chose the generic/general expert,
        # attenuate the domain-specific bases so finance/medical do not
        # both remain at high base values and compete with the generic choice.
        base_fin = base_fin * 0.7
        base_med = base_med * 0.7

    # Numeric weight scaling (saturating curve + type-aware)
    # We compute a saturating weight so each additional numeric mention has
    # diminishing returns: weight = base + (max - base) * (1 - exp(-alpha * effective_count))
    import math

    # Financial parameters (more aggressive)
    fin_base_min = 0.20
    fin_base_max = 0.75
    fin_alpha = 0.9

    # Medical and generic parameters (more conservative)
    med_base_min = 0.08
    med_base_max = 0.30
    med_alpha = 0.6

    gen_base_min = 0.06
    gen_base_max = 0.30
    gen_alpha = 0.5

    fin_numeric = fin_base_min + (fin_base_max - fin_base_min) * (1.0 - math.exp(-fin_alpha * effective_count))
    med_numeric = med_base_min + (med_base_max - med_base_min) * (1.0 - math.exp(-med_alpha * effective_count))
    gen_numeric = gen_base_min + (gen_base_max - gen_base_min) * (1.0 - math.exp(-gen_alpha * effective_count))

    # Divider: reduce numeric impact for very long text (normalize). Keep small for finance.
    fin_num_div = max(1.5, min(6.0, effective_count if effective_count > 0 else 1.5))
    med_num_div = max(2.0, min(8.0, effective_count if effective_count > 0 else 2.0))

    # Objectivity weight increases when subjectivity low
    fin_objectivity = 0.2 + (1.0 - subjectivity) * 0.15
    med_objectivity = 0.2 + (1.0 - subjectivity) * 0.15
    gen_subjectivity_weight = 0.25 * (1.0 - subjectivity) + 0.05

    # penalties
    actionable_pen = 0.05 + (0.05 if fin_recs else 0.0)
    clinical_pen = 0.12 + (0.08 if med_clinical else 0.0)

    # Transformer-friendly weights (coarse mapping)
    # For transformers we amplify numeric signals for financial models.
    tf_fin = {
        "numeric_weight": min(0.8, fin_numeric * 2.0),
        "subjectivity_weight": min(0.6, max(0.1, subjectivity * 1.0)),
        "actionable_weight": min(0.5, actionable_pen * 2.0),
    }
    tf_med = {
        "consult_weight": min(0.6, 0.25 + subjectivity * 0.8),
        "subjectivity_weight": min(0.6, max(0.05, subjectivity * 0.9)),
        "objective_weight": min(0.6, med_objectivity),
    }
    tf_gen = {
        "subjectivity_weight": min(0.6, gen_subjectivity_weight),
        "objective_weight": min(0.6, 0.25 + (1.0 - subjectivity) * 0.15),
        "numeric_weight": min(0.4, gen_numeric * 1.2),
    }

    # Apply transformer-level competitive bias: boost selected tf weights,
    # attenuate non-selected ones so transformer analyzers follow the same
    # expert preference as the rule-based analyzers.
    def _scale_tf(d: dict, mul: float, cap: float = 0.95):
        return {k: min(cap, float(v) * mul) for k, v in d.items()}

    if selected == "financial":
        tf_fin = _scale_tf(tf_fin, 1.25, cap=0.95)
        tf_med = _scale_tf(tf_med, 0.6, cap=0.95)
        tf_gen = _scale_tf(tf_gen, 0.7, cap=0.95)
    elif selected == "medical":
        tf_med = _scale_tf(tf_med, 1.25, cap=0.95)
        tf_fin = _scale_tf(tf_fin, 0.6, cap=0.95)
        tf_gen = _scale_tf(tf_gen, 0.7, cap=0.95)
    elif selected == "generic":
        tf_gen = _scale_tf(tf_gen, 1.15, cap=0.95)
        tf_fin = _scale_tf(tf_fin, 0.7, cap=0.95)
        tf_med = _scale_tf(tf_med, 0.7, cap=0.95)

    weights_fin = {
        "base": round(base_fin, 3),
        "numeric_weight": round(fin_numeric, 3),
        "numeric_divisor": round(fin_num_div, 3),
        "objectivity_weight": round(fin_objectivity, 3),
        "actionable_penalty": round(actionable_pen, 3),
    }
    weights_med = {
        "base": round(base_med, 3),
        "numeric_weight": round(med_numeric, 3),
        "numeric_divisor": round(med_num_div, 3),
        "objectivity_weight": round(med_objectivity, 3),
        "clinical_penalty": round(clinical_pen, 3),
    }
    weights_gen = {
        "base": round(base_gen, 3),
        "subjectivity_weight": round(gen_subjectivity_weight, 3),
    }

    return {
        "weights_fin": weights_fin,
        "weights_med": weights_med,
        "weights_gen": weights_gen,
        "tf_weights_fin": tf_fin,
        "tf_weights_med": tf_med,
        "tf_weights_gen": tf_gen,
        "meta": {
            "num_count": num_count,
            "subjectivity": round(subjectivity, 3),
            "polarity": round(polarity, 3),
            "agent": agent,
            "length": length,
        },
    }
