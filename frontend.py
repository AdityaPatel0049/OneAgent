import os
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper
from search_client import run_query, SearchClientError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import domain_analyzers as da

# Load .env in non-production environments to make local development easier.
try:
    from dotenv import load_dotenv
    if os.environ.get("ENV", "development") != "production":
        # load .env file if present (no-op if missing)
        load_dotenv()
except Exception:
    # dotenv is optional; if it's not installed, proceed without it
    pass

st.set_page_config(page_title="One Agent", page_icon="🔎", layout="centered")

def _load_css_from_static(fname: str):
    """Load a CSS file from the `static/` folder next to this script."""
    base = os.path.dirname(__file__)
    path = os.path.join(base, "static", fname)
    try:
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
    except Exception:
        # silently ignore missing CSS in case user hasn't created files
        pass

# Sidebar: theme toggle
st.sidebar.title("Appearance")
dark_mode = st.sidebar.checkbox("Dark mode", value=False)
# Provider selector (allow user to pick Cerebras or Serper for this session)
provider_options = ["cerebras", "serper"]
env_default = os.environ.get("SEARCH_PROVIDER", "cerebras").lower()
try:
    default_idx = provider_options.index(env_default) if env_default in provider_options else 0
except Exception:
    default_idx = 0
selected = st.sidebar.selectbox("Search provider", options=provider_options, index=default_idx, help="Choose which search provider to use for queries")
# persist selection to session and environment for consistency
st.session_state["selected_provider"] = selected
os.environ["SEARCH_PROVIDER"] = selected
# Provider status
def _provider_status() -> tuple[str, str]:
    """Return (provider_name, status_text) where status_text is a short status message."""
    provider = os.environ.get("SEARCH_PROVIDER", "cerebras").lower()
    if provider == "cerebras":
        api_url = os.environ.get("CEREBRAS_API_URL")
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_url:
            # not configured
            # fallback possibility
            if os.environ.get("SERPER_API_KEY"):
                return ("serper", "Cerebras not configured — falling back to Serper")
            return ("cerebras", "Cerebras not configured — set CEREBRAS_API_URL")
        if not api_key:
            return ("cerebras", "No API key set (CEREBRAS_API_KEY) — requests may fail")
        return ("cerebras", "Online")
    if provider == "serper":
        if os.environ.get("SERPER_API_KEY"):
            return ("serper", "Online")
        return ("serper", "No SERPER_API_KEY set")
    return (provider, "Unknown provider")


    

prov_name, prov_status = _provider_status()
st.sidebar.markdown(f"**Provider:** `{prov_name}` — {prov_status}")
# Quick action: auto-fill common Cerebras chat endpoint
if st.sidebar.button("Auto-fill Cerebras endpoint"):
    suggested = "https://api.cerebras.ai/v1/chat/completions"
    os.environ["CEREBRAS_API_URL"] = suggested
    st.sidebar.success(f"CEREBRAS_API_URL set to {suggested} (session)")
    # offer to persist to .env
    if st.sidebar.checkbox("Save this URL to project .env", key="save_cerebras_url"):
        try:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()

            # replace or append CEREBRAS_API_URL
            key = "CEREBRAS_API_URL"
            new_line = f"{key}={suggested}"
            found = False
            for idx, ln in enumerate(lines):
                if ln.strip().startswith(key + "="):
                    lines[idx] = new_line
                    found = True
                    break
            if not found:
                lines.append(new_line)

            with open(env_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else ""))

            st.sidebar.info(f"Wrote CEREBRAS_API_URL to {env_path}")
        except Exception as e:
            st.sidebar.error(f"Failed to write .env: {e}")
# Admin debug toggle (only in non-production)
is_admin = os.environ.get("ENV", "development") != "production"
show_debug = False
if is_admin:
    show_debug = st.sidebar.checkbox("Show provider debug log (admin only)", value=False, key="show_debug")
# Credentials persistence (optional): let users save keys and endpoint to a local .env
st.sidebar.markdown("---")
st.sidebar.markdown("### Credentials (optional)")
cerebras_key = st.sidebar.text_input("Cerebras API Key", value=os.environ.get("CEREBRAS_API_KEY", ""), type="password", key="cerebras_key_input")
cerebras_url = st.sidebar.text_input("Cerebras API URL", value=os.environ.get("CEREBRAS_API_URL", ""), key="cerebras_url_input")
cerebras_model = st.sidebar.text_input("Cerebras Model", value=os.environ.get("CEREBRAS_MODEL", ""), key="cerebras_model_input")
st.sidebar.caption("The .env file is written to the project folder. Do NOT commit .env to source control.")
serper_key = st.sidebar.text_input("Serper API Key", value=os.environ.get("SERPER_API_KEY", ""), type="password", key="serper_key_input")

# Transformer analyzer toggle: opt-in since it requires heavy deps
use_transformers = False
try:
    use_transformers = st.sidebar.checkbox("Use transformer analyzers (may require additional packages)", value=False)
except Exception:
    use_transformers = False

# Hugging Face Inference API settings (optional)
st.sidebar.markdown("---")
st.sidebar.markdown("### Hugging Face (optional)")
# Advanced toggle: hide/show sensitive token input
advanced_show_tokens = False
try:
    advanced_show_tokens = st.sidebar.checkbox("Advanced: show tokens", value=False, key="show_tokens_checkbox", help="Show sensitive token input fields (risky)")
except Exception:
    advanced_show_tokens = False

# Token value from session/env (kept hidden unless advanced_show_tokens)
hf_token_default = st.session_state.get("hf_token", os.environ.get("HF_INFERENCE_API_TOKEN", ""))
if advanced_show_tokens:
    hf_token_input = st.sidebar.text_input("HF Inference API Token", value=hf_token_default, type="password", key="hf_token_input")
else:
    # don't show the token field; show a status indicator instead
    hf_token_input = hf_token_default
    if hf_token_default:
        st.sidebar.markdown("**HF token:** ✅ configured (hidden)")
    else:
        st.sidebar.markdown("**HF token:** not configured")

# Model slugs (non-sensitive)
hf_fin_model_input = st.sidebar.text_input("HF Financial Model", value=os.environ.get("HF_FINANCIAL_MODEL", ""), key="hf_fin_model_input")
hf_med_model_input = st.sidebar.text_input("HF Medical Model", value=os.environ.get("HF_MEDICAL_MODEL", ""), key="hf_med_model_input")
hf_gen_model_input = st.sidebar.text_input("HF Generic Model", value=os.environ.get("HF_GENERIC_MODEL", ""), key="hf_gen_model_input")

# Persist to runtime environment so analyzer modules can pick them up immediately
try:
    if hf_token_input:
        os.environ["HF_INFERENCE_API_TOKEN"] = hf_token_input
        st.session_state["hf_token"] = hf_token_input
    if hf_fin_model_input:
        os.environ["HF_FINANCIAL_MODEL"] = hf_fin_model_input
        st.session_state["hf_financial_model"] = hf_fin_model_input
    if hf_med_model_input:
        os.environ["HF_MEDICAL_MODEL"] = hf_med_model_input
        st.session_state["hf_medical_model"] = hf_med_model_input
    if hf_gen_model_input:
        os.environ["HF_GENERIC_MODEL"] = hf_gen_model_input
        st.session_state["hf_generic_model"] = hf_gen_model_input
except Exception:
    pass

# Analysis weight sliders (make scoring configurable)
with st.sidebar.expander("Analysis weights (advanced)", expanded=False):
    st.markdown("**Financial analyzer weights**")
    fin_base = st.slider("Financial base", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="w_fin_base")
    fin_numeric = st.slider("Financial numeric weight", min_value=0.0, max_value=1.0, value=0.15, step=0.01, key="w_fin_numeric")
    fin_numeric_div = st.number_input("Financial numeric divisor", min_value=1.0, max_value=20.0, value=5.0, step=1.0, key="w_fin_numdiv")
    fin_objectivity = st.slider("Financial objectivity weight", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="w_fin_obj")
    fin_actionable_pen = st.slider("Financial actionable penalty", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="w_fin_pen")

    st.markdown("**Medical analyzer weights**")
    med_base = st.slider("Medical base", min_value=0.0, max_value=1.0, value=0.6, step=0.01, key="w_med_base")
    med_numeric = st.slider("Medical numeric weight", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="w_med_numeric")
    med_numeric_div = st.number_input("Medical numeric divisor", min_value=1.0, max_value=10.0, value=3.0, step=1.0, key="w_med_numdiv")
    med_objectivity = st.slider("Medical objectivity weight", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="w_med_obj")
    med_clinical_pen = st.slider("Medical clinical penalty", min_value=0.0, max_value=1.0, value=0.15, step=0.01, key="w_med_pen")

    st.markdown("**Generic analyzer weights**")
    gen_base = st.slider("Generic base", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="w_gen_base")
    gen_subjectivity = st.slider("Generic subjectivity weight", min_value=0.0, max_value=1.0, value=0.25, step=0.01, key="w_gen_subj")

    # transformer-style weights (for transformer analyzers)
    st.markdown("**Transformer weight helpers**")
    tf_fin_numeric = st.slider("TF Financial numeric weight", min_value=0.0, max_value=1.0, value=0.45, step=0.01, key="w_tf_fin_num")
    tf_fin_subjectivity = st.slider("TF Financial subjectivity weight", min_value=0.0, max_value=1.0, value=0.35, step=0.01, key="w_tf_fin_subj")
    tf_fin_actionable = st.slider("TF Financial actionable weight", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="w_tf_fin_act")
    tf_med_consult = st.slider("TF Medical consult weight", min_value=0.0, max_value=1.0, value=0.4, step=0.01, key="w_tf_med_consult")
    tf_med_subjectivity = st.slider("TF Medical subjectivity weight", min_value=0.0, max_value=1.0, value=0.3, step=0.01, key="w_tf_med_subj")
    tf_med_objective = st.slider("TF Medical objective weight", min_value=0.0, max_value=1.0, value=0.3, step=0.01, key="w_tf_med_obj")
    tf_gen_subj = st.slider("TF Generic subjectivity weight", min_value=0.0, max_value=1.0, value=0.4, step=0.01, key="w_tf_gen_subj")
    tf_gen_obj = st.slider("TF Generic objective weight", min_value=0.0, max_value=1.0, value=0.4, step=0.01, key="w_tf_gen_obj")
    tf_gen_num = st.slider("TF Generic numeric weight", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="w_tf_gen_num")

    # assemble weights dicts for passing into analyzers
    weights_fin = {"base": fin_base, "numeric_weight": fin_numeric, "numeric_divisor": fin_numeric_div, "objectivity_weight": fin_objectivity, "actionable_penalty": fin_actionable_pen}
    weights_med = {"base": med_base, "numeric_weight": med_numeric, "numeric_divisor": med_numeric_div, "objectivity_weight": med_objectivity, "clinical_penalty": med_clinical_pen}
    weights_gen = {"base": gen_base, "subjectivity_weight": gen_subjectivity}
    tf_weights_fin = {"numeric_weight": tf_fin_numeric, "subjectivity_weight": tf_fin_subjectivity, "actionable_weight": tf_fin_actionable}
    tf_weights_med = {"consult_weight": tf_med_consult, "subjectivity_weight": tf_med_subjectivity, "objective_weight": tf_med_objective}
    tf_weights_gen = {"subjectivity_weight": tf_gen_subj, "objective_weight": tf_gen_obj, "numeric_weight": tf_gen_num}
if st.sidebar.button("Save credentials to .env"):
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

        def upsert(key: str, val: str):
            new_line = f"{key}={val}"
            found = False
            for idx, ln in enumerate(lines):
                if ln.strip().startswith(key + "="):
                    lines[idx] = new_line
                    found = True
                    break
            if not found:
                lines.append(new_line)

        if cerebras_key:
            upsert("CEREBRAS_API_KEY", cerebras_key)
            os.environ["CEREBRAS_API_KEY"] = cerebras_key
        if cerebras_url:
            upsert("CEREBRAS_API_URL", cerebras_url)
            os.environ["CEREBRAS_API_URL"] = cerebras_url
        if cerebras_model:
            upsert("CEREBRAS_MODEL", cerebras_model)
            os.environ["CEREBRAS_MODEL"] = cerebras_model
        if serper_key:
            upsert("SERPER_API_KEY", serper_key)
            os.environ["SERPER_API_KEY"] = serper_key
        # Hugging Face values
        if hf_token_input:
            upsert("HF_INFERENCE_API_TOKEN", hf_token_input)
            os.environ["HF_INFERENCE_API_TOKEN"] = hf_token_input
        if hf_fin_model_input:
            upsert("HF_FINANCIAL_MODEL", hf_fin_model_input)
            os.environ["HF_FINANCIAL_MODEL"] = hf_fin_model_input
        if hf_med_model_input:
            upsert("HF_MEDICAL_MODEL", hf_med_model_input)
            os.environ["HF_MEDICAL_MODEL"] = hf_med_model_input
        if hf_gen_model_input:
            upsert("HF_GENERIC_MODEL", hf_gen_model_input)
            os.environ["HF_GENERIC_MODEL"] = hf_gen_model_input

        # write back file
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else ""))

        st.sidebar.success("Saved provided credentials to .env — do not commit this file.")
    except Exception as e:
        st.sidebar.error(f"Failed to save .env: {e}")
# Load chosen CSS
if dark_mode:
    _load_css_from_static("frontend_dark.css")
else:
    _load_css_from_static("frontend_light.css")

st.title("🔎 One Agent")
st.markdown("<div style='margin-top:-8px;margin-bottom:6px;color:var(--secondary, #6b7280)'>A compact interface for running expert agents and analyzing their responses.</div>", unsafe_allow_html=True)

if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.environ.get("SERPER_API_KEY", "")
if "history" not in st.session_state:
    st.session_state["history"] = []

# Sentiment analyzer (VADER)
analyzer = SentimentIntensityAnalyzer()

# Agent presets (moved to module-level so rendering can reuse colors)
AGENTS = {
    "General": {
        "prefix": "",
        "description": "A general-purpose assistant that returns concise, factual answers.",
        "color": "#7f8c8d"
    },
    "Financial Expert": {
        "prefix": "You are a financial expert. Answer concisely with financial considerations, risks, and citations when possible.",
        "description": "Provides finance-focused answers and highlights risks. Not financial advice.",
        "color": "#16a085"
    },
    "Medical Expert": {
        "prefix": "You are a medical expert. Provide evidence-based, cautious information and recommend consulting a licensed professional when appropriate.",
        "description": "Provides medical information; NOT a substitute for professional medical advice.",
        "color": "#c0392b"
    },
    "Technology Expert": {
        "prefix": "You are a technology expert. Focus on technical accuracy, relevant standards, and succinct implementation guidance.",
        "description": "Technical and developer-focused answers.",
        "color": "#2980b9"
    },
    "Legal Expert": {
        "prefix": "You are a legal expert. Provide general legal information, note jurisdiction differences, and advise consulting a lawyer for binding advice.",
        "description": "General legal information only; not legal advice.",
        "color": "#8e44ad"
    }
}

# which agents are considered sensitive (don't show internal prefix)
SENSITIVE_AGENTS = {"Medical Expert", "Legal Expert", "Financial Expert"}


def render_sentiment_bar(compound: float) -> str:
    """Return an HTML snippet for a horizontal sentiment bar for a compound score in [-1, 1]."""
    # clamp just in case
    if compound is None:
        compound = 0.0
    compound = max(-1.0, min(1.0, float(compound)))
    # map -1..1 to 0..100
    percent = int(round((compound + 1) / 2 * 100))

    # choose color for bar and label
    if compound >= 0.05:
        color = "#2ecc71"  # green
        label_color = "#0b6e2b"
        text_label = "Positive"
    elif compound <= -0.05:
        color = "#e74c3c"  # red
        label_color = "#7a1f1a"
        text_label = "Negative"
    else:
        color = "#95a5a6"  # gray
        label_color = "#5a6368"
        text_label = "Neutral"

    # HTML for a compact bar with colored label
    html = f"""
<div style="display:flex;align-items:center;gap:10px;font-family:Inter,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <div style="flex:1;background:#eeeeee;border-radius:10px;height:14px;overflow:hidden;box-shadow:inset 0 1px 2px rgba(0,0,0,0.06)">
    <div style="width:{percent}%;background:{color};height:100%;border-radius:10px"></div>
  </div>
  <div style="min-width:140px;text-align:right;font-size:13px;color:#222">{compound:+.2f} &nbsp; <span style='font-weight:700;color:{label_color};padding:4px 8px;border-radius:8px;background:rgba(0,0,0,0.03)'> {text_label} </span></div>
</div>
"""
    return html


def render_bias_objectivity(subjectivity: float, bias_score: float, objectivity: float, bias_label: str) -> str:
    """Return HTML showing objectivity percentage and a colored bias label pill."""
    try:
        subj_pct = int(round(max(0.0, min(1.0, float(subjectivity))) * 100))
    except Exception:
        subj_pct = 0
    try:
        obj_pct = int(round(max(0.0, min(1.0, float(objectivity))) * 100))
    except Exception:
        obj_pct = 100

    # bias color
    if bias_label == "High":
        bias_color = "#c0392b"  # deep red
    elif bias_label == "Medium":
        bias_color = "#e67e22"  # orange
    else:
        bias_color = "#2ecc71"  # green

    html = f"""
<div style="display:flex;gap:12px;align-items:center;font-size:13px;margin-top:6px">
  <div style="color:#333">Objectivity: <strong>{obj_pct}%</strong></div>
  <div style="color:#333">Subjectivity: <strong>{subj_pct}%</strong></div>
  <div style="margin-left:auto"><span style="background:{bias_color};color:#fff;padding:6px 10px;border-radius:999px;font-weight:700">{bias_label} bias</span></div>
</div>
"""
    return html

with st.form("search_form"):
    api_key_input = st.text_input("Serper API Key", value=st.session_state["api_key"], type="password")
    agent = st.selectbox("Agent", options=list(AGENTS.keys()), index=0, help="Choose an expert persona to tailor the response")
    agent_description = AGENTS.get(agent, {}).get("description", "")
    agent_color = AGENTS.get(agent, {}).get("color", "#777")
    if agent_description:
        st.caption(agent_description)
        # show a small colored pill next to description
        pill_html = f"<div style='margin-top:6px'><span style='background:{agent_color};color:#fff;padding:6px 10px;border-radius:999px;font-weight:700'>{agent}</span></div>"
        st.markdown(pill_html, unsafe_allow_html=True)
    query = st.text_input("Search query", value="")
    submitted = st.form_submit_button("Search")

    if submitted:
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            if api_key_input:
                os.environ["SERPER_API_KEY"] = api_key_input
                st.session_state["api_key"] = api_key_input

            try:
                with st.spinner("Running search..."):
                    # prepend agent prefix to the query so persona affects search output
                    prefix = AGENTS.get(agent, {}).get("prefix", "")
                    full_query = f"{prefix}\n\n{query}" if prefix else query
                    try:
                        provider_to_use = st.session_state.get("selected_provider", os.environ.get("SEARCH_PROVIDER"))
                        result = run_query(full_query, provider=provider_to_use)
                    except SearchClientError as sce:
                        st.error("Search provider error: " + str(sce))
                        st.info("Check `CEREBRAS_API_URL` / `CEREBRAS_API_KEY` or set `SEARCH_PROVIDER=serper`")
                        # If admin has enabled debug, show attempt details captured by the client
                        try:
                            if show_debug and hasattr(sce, 'attempts') and sce.attempts:
                                st.subheader("Provider attempts")
                                for a in sce.attempts:
                                    url = a.get('url')
                                    status = a.get('status')
                                    snippet = a.get('snippet')
                                    st.markdown(f"- `{url}` — status: `{status}` — `{snippet}`")
                        except Exception:
                            # don't let debug display break the flow
                            pass
                        # mark as failed and skip analysis/storage
                        result = None

                # perform sentiment analysis on the result (skip if provider failed)
                try:
                    if result is None:
                        # provider call failed earlier; skip analysis and storing
                        continue_flag = True
                    else:
                        continue_flag = False
                        text = result if isinstance(result, str) else str(result)

                    # VADER sentiment (existing)
                    scores = analyzer.polarity_scores(text)
                    compound = scores.get("compound", 0.0)
                    if compound >= 0.05:
                        label = "Positive"
                    elif compound <= -0.05:
                        label = "Negative"
                    else:
                        label = "Neutral"

                    # Domain-specific analysis: prefer transformer analyzers if enabled,
                    # otherwise fall back to the heuristic analyzers in `domain_analyzers`.
                    try:
                        domain_info = None
                        if use_transformers:
                            try:
                                import domain_analyzers_transformers as datf
                                if agent == "Financial Expert":
                                    domain_info = datf.analyze_financial(text, weights=tf_weights_fin)
                                elif agent == "Medical Expert":
                                    domain_info = datf.analyze_medical(text, weights=tf_weights_med)
                                else:
                                    domain_info = datf.analyze_generic(text, weights=tf_weights_gen)
                            except Exception:
                                # transformer analysers not available or failed; fall back
                                domain_info = None

                        if domain_info is None:
                            # use rule-based analyzer (pass UI-configured weights)
                            if agent == "Financial Expert":
                                domain_info = da.analyze_financial(text, weights=weights_fin)
                            elif agent == "Medical Expert":
                                domain_info = da.analyze_medical(text, weights=weights_med)
                            else:
                                domain_info = da.analyze_generic(text, weights=weights_gen)

                        # pull basic signal values from domain_info when available
                        polarity = domain_info.get("polarity", 0.0)
                        subjectivity = domain_info.get("subjectivity", 0.0)
                        objectivity = max(0.0, 1.0 - subjectivity)
                        domain_score = domain_info.get("score")
                    except Exception:
                        # fallback to TextBlob if domain analyzer fails entirely
                        blob = TextBlob(text)
                        polarity = float(blob.sentiment.polarity)
                        subjectivity = float(blob.sentiment.subjectivity)
                        objectivity = max(0.0, 1.0 - subjectivity)
                        domain_score = None

                    # Basic bias heuristic (kept for backwards compatibility)
                    try:
                        bias_score = abs(float(polarity)) * float(subjectivity)
                        if bias_score >= 0.4:
                            bias_label = "High"
                        elif bias_score >= 0.2:
                            bias_label = "Medium"
                        else:
                            bias_label = "Low"
                    except Exception:
                        bias_score = 0.0
                        bias_label = "Low"
                except Exception:
                    scores = {"compound": 0.0}
                    compound = 0.0
                    label = "Unknown"
                    polarity = 0.0
                    subjectivity = 0.0
                    objectivity = 1.0
                    bias_score = 0.0
                    bias_label = "Low"

                # store in history (most recent first) if provider call succeeded
                if not ("continue_flag" in locals() and continue_flag):
                    # include the domain analysis dict if present
                    domain_info = locals().get("domain_info") if "domain_info" in locals() else None
                    st.session_state["history"].insert(0, {"query": query, "sent_query": full_query, "agent": agent, "result": result, "sentiment": {"label": label, "compound": compound, "details": scores, "polarity": polarity, "subjectivity": subjectivity, "objectivity": objectivity, "bias_score": bias_score, "bias_label": bias_label}, "domain_analysis": domain_info})
                    st.success("Search complete")
                else:
                    st.info("Search not saved due to provider error.")
            except Exception as e:
                st.error(f"Error running search: {e}")

st.markdown("---")

if st.session_state["history"]:
    st.subheader("Recent searches")
    for i, item in enumerate(st.session_state["history"]):
        with st.expander(f"{i+1}. {item['query']}"):
            # show which agent was used with colored pill and color the result block
            agent_used = item.get("agent", "General")
            agent_color = AGENTS.get(agent_used, {}).get("color", "#777")
            pill_html = f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'><span class='agent-pill' style='background:{agent_color};'>{agent_used}</span></div>"
            st.markdown(pill_html, unsafe_allow_html=True)
            # show the actual query that was sent (with agent prefix)
            sent_query = item.get("sent_query")
            if sent_query:
                st.markdown("**Sent query:**")
                # obfuscate internal agent prompt for sensitive agents
                if agent_used in SENSITIVE_AGENTS:
                    # show an obfuscated indicator and the original user query only
                    user_query = item.get("query", "")
                    obf = "[AGENT PREFIX REMOVED FOR PRIVACY]\n\n" + user_query
                    st.code(obf, language="")
                else:
                    st.code(sent_query, language="")
            # render the result inside a left-colored border for quick scanning
            try:
                result_text = item["result"] if isinstance(item["result"], str) else str(item["result"])
                result_html = f"<div class='result-card' style='border-left:6px solid {agent_color};'>{result_text}</div>"
                st.markdown(result_html, unsafe_allow_html=True)
            except Exception:
                st.markdown("<div style='color:#c0392b'>Failed to render result</div>", unsafe_allow_html=True)
            # show domain-specific analysis if present
            domain_info = item.get("domain_analysis")
            if domain_info:
                try:
                    # compact domain card
                    dname = domain_info.get("domain", "domain")
                    dscore = domain_info.get('score')
                    highlights = domain_info.get("highlights", []) or []
                    issues = domain_info.get("issues", []) or []
                    recs = domain_info.get("recommendations", []) or []
                    card_lines = [f"<div class='domain-card'><strong>Domain:</strong> <code>{dname}</code><br/><strong>Score:</strong> <code>{dscore}</code>"]
                    if highlights:
                        card_lines.append(f"<div style='margin-top:6px'><strong>Highlights:</strong> {', '.join(highlights)}</div>")
                    if issues:
                        card_lines.append(f"<div style='margin-top:6px;color:#c0392b'><strong>Issues:</strong> {'; '.join(issues)}</div>")
                    if recs:
                        card_lines.append(f"<div style='margin-top:6px;color:#16a085'><strong>Recommendations:</strong> {'; '.join(recs)}</div>")
                    card_lines.append("</div>")
                    st.markdown('\n'.join(card_lines), unsafe_allow_html=True)
                except Exception:
                    st.markdown("<div style='color:#e67e22'>Failed to render domain analysis</div>", unsafe_allow_html=True)
                
            # Show sentiment
            sentiment = item.get("sentiment", {})
            label = sentiment.get("label", "Unknown")
            compound = sentiment.get("compound", 0.0)
            cols = st.columns([1, 4])
            with cols[0]:
                if st.button("Copy result", key=f"copy_{i}"):
                    st.experimental_set_query_params(_copied=item["query"])  # small UX trick
            with cols[1]:
                # render visual sentiment bar
                try:
                    bar_html = render_sentiment_bar(compound)
                    st.markdown(bar_html, unsafe_allow_html=True)
                    # show bias/objectivity details if present
                    subj = sentiment.get("subjectivity", None)
                    bias_score = sentiment.get("bias_score", None)
                    obj = sentiment.get("objectivity", None)
                    bias_label = sentiment.get("bias_label", "Low")
                    if subj is not None:
                        details_html = render_bias_objectivity(subj, bias_score, obj, bias_label)
                        st.markdown(details_html, unsafe_allow_html=True)
                except Exception:
                    st.markdown(f"**Sentiment:** {label}  —  `compound={compound:.2f}`")

    if st.button("Clear history"):
        st.session_state["history"] = []
else:
    st.info("No searches yet — enter a query above to get started.")

st.markdown(
    """
    ---
    **Tips:**
    - Set `SERPER_API_KEY` in your environment to avoid entering it each time.
    - The key can also be entered in the top field (kept in the session only).
    """
)
