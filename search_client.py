import os
import json
from typing import Optional

# Try to import Serper wrapper as a fallback
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
except Exception:
    GoogleSerperAPIWrapper = None

import requests
import urllib.parse
try:
    from dotenv import load_dotenv
    if os.environ.get("ENV", "development") != "production":
        load_dotenv()
except Exception:
    pass


class SearchClientError(Exception):
    def __init__(self, message: str, attempts: Optional[list] = None):
        super().__init__(message)
        self.attempts = attempts or []

    def __str__(self):
        base = super().__str__()
        return base


def run_query(query: str, provider: Optional[str] = None) -> str:
    """Run a query using the selected provider.

    Provider selection order:
    - explicit `provider` argument if provided
    - environment variable `SEARCH_PROVIDER` if set
    - default to 'cerebras'

    For `cerebras`, the adapter expects the following environment variables to be set:
    - `CEREBRAS_API_URL` (HTTP endpoint that accepts POST with JSON {"prompt": "..."})
    - `CEREBRAS_API_KEY` (optional, used as Bearer token)

    The adapter sends JSON {"prompt": query} and expects a JSON response with a top-level
    `text` or `result` field containing the generated answer. If that field is missing,
    it will return the full JSON as a string.

    If `cerebras` is not configured, this function will fall back to Serper (if available).
    """
    if provider is None:
        provider = os.environ.get("SEARCH_PROVIDER", "cerebras")

    provider = provider.lower()

    if provider == "cerebras":
        api_url = os.environ.get("CEREBRAS_API_URL")
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_url:
            # fallback to serper if cerebras not configured
            if GoogleSerperAPIWrapper is not None:
                provider = "serper"
            else:
                raise SearchClientError("CEREBRAS_API_URL is not set and Serper wrapper not available")

    if provider == "cerebras":
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Support both a simple prompt-style endpoint and Chat Completions style endpoints
        try:
            # If the URL looks like a chat completions endpoint, send messages
            if "/chat/completions" in api_url:
                model = os.environ.get("CEREBRAS_MODEL", "cerebras-chat")
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": query}]
                }
            else:
                # fallback generic prompt-based API
                payload = {"prompt": query}

            attempts = []

            def _post_and_parse(url: str):
                try:
                    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
                    status = r.status_code
                    text = None
                    try:
                        data = r.json()
                        text = json.dumps(data) if not isinstance(data, str) else data
                    except ValueError:
                        data = r.text
                        text = data[:200]
                    attempts.append({"url": url, "status": status, "snippet": (text if text else "")})
                    r.raise_for_status()
                    return True, data
                except requests.HTTPError as he:
                    status = he.response.status_code if (hasattr(he, 'response') and he.response is not None) else None
                    snippet = None
                    try:
                        snippet = he.response.text[:200] if (hasattr(he, 'response') and he.response is not None) else str(he)
                    except Exception:
                        snippet = str(he)
                    attempts.append({"url": url, "status": status, "snippet": snippet, "error": str(he)})
                    raise
                except requests.RequestException as re:
                    attempts.append({"url": url, "status": None, "snippet": str(re), "error": str(re)})
                    raise

            # first attempt
            try:
                ok, data = _post_and_parse(api_url)
            except requests.HTTPError as http_err:
                # if 404, try common alternative endpoints under the service root (domain)
                if hasattr(http_err, 'response') and http_err.response is not None and http_err.response.status_code == 404:
                    parsed = urllib.parse.urlparse(api_url)
                    if parsed.scheme and parsed.netloc:
                        root = f"{parsed.scheme}://{parsed.netloc}"
                    else:
                        # fallback to trimming path segments
                        root = api_url.split('/v1')[0].rstrip('/')

                    alt_paths = ['/v1/chat/completions', '/v1/generate', '/v1/completions']
                    last_exc = http_err
                    for p in alt_paths:
                        try:
                            alt_url = root + p
                            ok, data = _post_and_parse(alt_url)
                            # if succeeded, return parsed data
                            break
                        except requests.HTTPError as he:
                            last_exc = he
                            continue
                    else:
                        # none of the alternatives worked
                        raise SearchClientError(f"Cerebras request failed (404). Tried common endpoints under {root}", attempts=attempts) from last_exc
                else:
                    raise SearchClientError(f"Cerebras request failed: {http_err}", attempts=attempts)
            except requests.RequestException as e:
                raise SearchClientError(f"Cerebras request failed: {e}", attempts=attempts)

            # 'data' now contains either a parsed JSON or raw text

            # Chat-style responses (choices -> message -> content)
            if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and len(data["choices"])>0:
                first = data["choices"][0]
                # common shapes
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        return first["message"]["content"]
                    if "text" in first:
                        return first["text"]
                    if "content" in first:
                        return first["content"]

            # look for common top-level fields
            for key in ("text", "result", "output", "answer"):
                if key in data:
                    return data[key]

            # fallback: return the whole JSON stringified
            return json.dumps(data)
        except requests.RequestException as e:
            raise SearchClientError(f"Cerebras request failed: {e}")

    # fallback to Serper
    if provider == "serper":
        if GoogleSerperAPIWrapper is None:
            raise SearchClientError("Serper wrapper unavailable")
        try:
            search = GoogleSerperAPIWrapper()
            return search.run(query)
        except Exception as e:
            raise SearchClientError(f"Serper search failed: {e}")

    raise SearchClientError(f"Unknown provider: {provider}")
