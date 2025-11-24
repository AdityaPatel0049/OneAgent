import os
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper
from search_client import run_query, SearchClientError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import domain_analyzers as da
import backend_auto_weights as baw
import math

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

# Advanced toggle (floating): use query param `advanced=1` to show advanced settings
# Use the stable `st.query_params` API (replaces experimental_get_query_params)
params = st.query_params
show_advanced = params.get("advanced", ["0"])[0] == "1"
toggle_href = "?advanced=0" if show_advanced else "?advanced=1"
toggle_label = "Hide advanced" if show_advanced else "Show advanced"
# small floating pill in the top-right
toggle_html = f"""
<div style="position:fixed;top:12px;right:12px;z-index:9999">
    <a href="{toggle_href}" style="text-decoration:none;display:inline-block;background:rgba(0,0,0,0.6);color:#fff;padding:8px 12px;border-radius:20px;font-weight:700;font-family:Inter,Segoe UI,Roboto,Helvetica,Arial,sans-serif">{toggle_label}</a>
</div>
"""
st.markdown(toggle_html, unsafe_allow_html=True)

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
# Admin debug toggle (only in non-production). Render only in advanced mode.
is_admin = os.environ.get("ENV", "development") != "production"
show_debug = False
if is_admin and show_advanced:
    show_debug = st.sidebar.checkbox("Show provider debug log (admin only)", value=False, key="show_debug")
# Credentials persistence (optional): let users save keys and endpoint to a local .env
if show_advanced:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Credentials (optional)")
    cerebras_key = st.sidebar.text_input("Cerebras API Key", value=os.environ.get("CEREBRAS_API_KEY", ""), type="password", key="cerebras_key_input")
    cerebras_url = st.sidebar.text_input("Cerebras API URL", value=os.environ.get("CEREBRAS_API_URL", ""), key="cerebras_url_input")
    cerebras_model = st.sidebar.text_input("Cerebras Model", value=os.environ.get("CEREBRAS_MODEL", ""), key="cerebras_model_input")
    st.sidebar.caption("The .env file is written to the project folder. Do NOT commit .env to source control.")
    serper_key = st.sidebar.text_input("Serper API Key", value=os.environ.get("SERPER_API_KEY", ""), type="password", key="serper_key_input")
else:
    # keep placeholders so later code that references these variables doesn't fail
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
    cerebras_url = os.environ.get("CEREBRAS_API_URL", "")
    cerebras_model = os.environ.get("CEREBRAS_MODEL", "")
    serper_key = os.environ.get("SERPER_API_KEY", "")

# Transformer analyzer toggle: opt-in since it requires heavy deps. Show in advanced only.
use_transformers = False
if show_advanced:
    try:
        use_transformers = st.sidebar.checkbox("Use transformer analyzers (may require additional packages)", value=False)
    except Exception:
        use_transformers = False
else:
    use_transformers = False

# Hugging Face Inference API settings (optional) — shown only in advanced mode
if show_advanced:
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
else:
    # defaults when advanced hidden
    hf_token_input = st.session_state.get("hf_token", os.environ.get("HF_INFERENCE_API_TOKEN", ""))
    hf_fin_model_input = os.environ.get("HF_FINANCIAL_MODEL", "")
    hf_med_model_input = os.environ.get("HF_MEDICAL_MODEL", "")
    hf_gen_model_input = os.environ.get("HF_GENERIC_MODEL", "")

# Analysis weights are now managed automatically by the backend; UI weight controls removed.
with st.sidebar.expander("Analysis weights (managed automatically)", expanded=False):
    st.markdown("Domain analyzer weights are managed by the backend. No manual controls are exposed.")
if show_advanced and st.sidebar.button("Save credentials to .env"):
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

hero_html = """
<div class='hero'>
    <div class='hero-left'>
        <div class='hero-logo'>🔎</div>
        <div class='hero-title'>One Agent</div>
        <div class='hero-sub'>A compact interface for running expert agents and analyzing their responses.</div>
    </div>
    <div class='hero-right'>
        <div class='hero-cta'>
            <div class='cta-note'>Try a sample query or pick an expert persona to get started.</div>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

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


def render_domain_scores(scores: dict) -> str:
    """Return a small markdown string listing domain/analyzer scores.

    Expects a dict mapping analyzer name -> numeric score. Robust to missing
    values and formats the output succinctly for the history pane.
    """
    if not scores:
        return ""
    lines = ["**All domain scores:**"]
    for name, val in scores.items():
        try:
            if val is None:
                lines.append(f"- **{name.title()}**: `-`")
            else:
                lines.append(f"- **{name.title()}**: `{float(val):.3f}`")
        except Exception:
            lines.append(f"- **{name.title()}**: `{val}`")
    return "\n".join(lines)

with st.form("search_form"):
    # Serper API Key input: hide by default unless advanced mode is enabled
    if show_advanced:
        api_key_input = st.text_input("Serper API Key", value=st.session_state["api_key"], type="password")
    else:
        # show a compact status indicator instead of the input field
        existing_key = st.session_state.get("api_key", os.environ.get("SERPER_API_KEY", ""))
        tooltip = "Hidden for privacy. Toggle the floating 'Show advanced' pill (top-right) or append ?advanced=1 to the URL to reveal and edit keys."
        try:
            if existing_key:
                st.markdown(f"**Serper API Key:** ✅ configured (hidden) <span title=\"{tooltip}\" style=\"color:#6b7280;margin-left:8px\">ℹ️</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Serper API Key:** not configured <span title=\"{tooltip}\" style=\"color:#6b7280;margin-left:8px\">ℹ️</span>", unsafe_allow_html=True)
        except Exception:
            # fallback to plain caption if markup fails
            if existing_key:
                st.caption("Serper API Key: ✅ configured (hidden)")
            else:
                st.caption("Serper API Key: not configured")
        api_key_input = existing_key
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
                        # compute automatic backend weights for this query
                        try:
                            auto = baw.auto_weights(text, agent, use_transformers=use_transformers)
                        except Exception:
                            auto = None

                        if use_transformers:
                            try:
                                import domain_analyzers_transformers as datf
                                if agent == "Financial Expert":
                                    domain_info = datf.analyze_financial(text, weights=(auto.get("tf_weights_fin") if auto else None))
                                elif agent == "Medical Expert":
                                    domain_info = datf.analyze_medical(text, weights=(auto.get("tf_weights_med") if auto else None))
                                else:
                                    domain_info = datf.analyze_generic(text, weights=(auto.get("tf_weights_gen") if auto else None))
                            except Exception:
                                # transformer analysers not available or failed; fall back
                                domain_info = None

                        if domain_info is None:
                            # use rule-based analyzer with backend-computed weights when available
                            if agent == "Financial Expert":
                                domain_info = da.analyze_financial(text, weights=(auto.get("weights_fin") if auto else None))
                            elif agent == "Medical Expert":
                                domain_info = da.analyze_medical(text, weights=(auto.get("weights_med") if auto else None))
                            else:
                                domain_info = da.analyze_generic(text, weights=(auto.get("weights_gen") if auto else None))

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
                    # compute and persist per-analyzer scores at query time so rendering uses the same values
                    all_scores = {}
                    try:
                        text_for_scores = text if isinstance(text, str) else str(text)
                    except Exception:
                        text_for_scores = result if isinstance(result, str) else str(result)
                    try:
                        # compute auto-weights for this stored result as well
                        try:
                            auto2 = baw.auto_weights(text_for_scores, agent, use_transformers=use_transformers)
                        except Exception:
                            auto2 = None
                        try:
                            fin = da.analyze_financial(text_for_scores, weights=(auto2.get("weights_fin") if auto2 else None))
                            all_scores["financial"] = fin.get("score")
                        except Exception:
                            all_scores["financial"] = None
                        try:
                            med = da.analyze_medical(text_for_scores, weights=(auto2.get("weights_med") if auto2 else None))
                            all_scores["medical"] = med.get("score")
                        except Exception:
                            all_scores["medical"] = None
                        try:
                            gen = da.analyze_generic(text_for_scores, weights=(auto2.get("weights_gen") if auto2 else None))
                            all_scores["generic"] = gen.get("score")
                        except Exception:
                            all_scores["generic"] = None

                        if use_transformers:
                            try:
                                import domain_analyzers_transformers as datf
                                try:
                                    tf_fin = datf.analyze_financial(text_for_scores)
                                    all_scores["tf_financial"] = tf_fin.get("score")
                                except Exception:
                                    all_scores["tf_financial"] = None
                                try:
                                    tf_med = datf.analyze_medical(text_for_scores)
                                    all_scores["tf_medical"] = tf_med.get("score")
                                except Exception:
                                    all_scores["tf_medical"] = None
                                try:
                                    tf_gen = datf.analyze_generic(text_for_scores)
                                    all_scores["tf_generic"] = tf_gen.get("score")
                                except Exception:
                                    all_scores["tf_generic"] = None
                            except Exception:
                                # transformer analyzers not available; ignore
                                pass
                    except Exception:
                        # fallback: leave all_scores possibly incomplete
                        pass

                    # compute a normalized (softmax) distribution across the three main domains
                    try:
                        keys = ["financial", "medical", "generic"]
                        beta = 4.0
                        raw_vals = [max(0.0, float(all_scores.get(k) or 0.0)) for k in keys]
                        m = max(raw_vals) if raw_vals else 0.0
                        exps = [math.exp(beta * (v - m)) for v in raw_vals]
                        s = sum(exps) or 1.0
                        normalized = {keys[i]: round(exps[i] / s, 3) for i in range(len(keys))}
                    except Exception:
                        normalized = {}

                    st.session_state["history"].insert(0, {"query": query, "sent_query": full_query, "agent": agent, "result": result, "sentiment": {"label": label, "compound": compound, "details": scores, "polarity": polarity, "subjectivity": subjectivity, "objectivity": objectivity, "bias_score": bias_score, "bias_label": bias_label}, "domain_analysis": domain_info, "all_scores": all_scores, "all_scores_normalized": normalized})
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

                    # display stored per-analyzer scores computed at query time
                    # Prefer normalized scores (softmax across domains) if available,
                    # otherwise fall back to the raw per-analyzer scores.
                    try:
                        stored_scores_norm = item.get("all_scores_normalized", {}) or {}
                        stored_scores_raw = item.get("all_scores", {}) or {}
                        # choose which to display (normalized preferred)
                        stored_scores = stored_scores_norm if stored_scores_norm else stored_scores_raw
                        if stored_scores:
                            score_lines = ["<div class='score-grid' style='margin-top:8px'><strong style='width:100%'>All analyzer scores:</strong>"]
                            for k, v in stored_scores.items():
                                display_name = k.replace("_", " ").title()
                                try:
                                    sc_text = f"{float(v):.3f}" if v is not None else "-"
                                except Exception:
                                    sc_text = str(v)
                                score_lines.append(f"<div class='score-pill'><strong>{display_name}:</strong> <code>{sc_text}</code></div>")
                            score_lines.append("</div>")
                            card_lines.extend(score_lines)
                    except Exception:
                        pass

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
                    # modern Streamlit: assign to `st.query_params` to trigger rerun
                    st.query_params = {"_copied": [item["query"]]}
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
