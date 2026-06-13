"""Venice.ai backend — OpenAI-compatible chat completions API.

Venice.ai offers private, uncensored inference with vision-capable models.
Uses the OpenAI chat completions format with an API key from the node UI.
The key is never written into the workflow.
"""

import base64
import io as _io
import json

import requests

from .caption_schema import build_user_prompt, get_system_prompt, parse_caption

API_ROOT = "https://api.venice.ai/api/v1"

# Models from Venice docs — grouped by capability.
# Vision models support image input (required for reference images).
VENICE_MODELS = [
    # Vision-capable models (support image_url content)
    {"id": "qwen3-vl-235b-a22b", "display_name": "Qwen3-VL 235B (vision)", "vision": True},
    {"id": "llama-3.2-90b-vision", "display_name": "Llama 3.2 90B Vision", "vision": True},
    {"id": "llama-3.2-11b-vision", "display_name": "Llama 3.2 11B Vision", "vision": True},
    {"id": "pixtral-12b", "display_name": "Pixtral 12B (vision)", "vision": True},
    # Text-only models
    {"id": "zai-org-glm-5", "display_name": "GLM 5 (default)", "vision": False},
    {"id": "kimi-k2-6", "display_name": "Kimi K2 6 (reasoning)", "vision": False},
    {"id": "claude-opus-4-8", "display_name": "Claude Opus 4 (complex tasks)", "vision": False},
    {"id": "venice-uncensored-1-2", "display_name": "Venice Uncensored 1.2", "vision": False},
    {"id": "deepseek-r1-671b", "display_name": "DeepSeek R1 671B", "vision": False},
    {"id": "llama-3.3-70b", "display_name": "Llama 3.3 70B", "vision": False},
    {"id": "qwen-2.5-72b", "display_name": "Qwen 2.5 72B", "vision": False},
    {"id": "qwen-2.5-vl-72b", "display_name": "Qwen 2.5 VL 72B (vision)", "vision": True},
    {"id": "qwen-2.5-vl-7b", "display_name": "Qwen 2.5 VL 7B (vision)", "vision": True},
    {"id": "mistral-31-24b", "display_name": "Mistral 3.1 24B", "vision": False},
]


def list_models(api_key):
    """Return [{id, display_name}] for available Venice models.

    Tries the /v1/models endpoint first. If that fails for any reason,
    falls back to the hardcoded list from Venice documentation.
    """
    if not api_key:
        raise ValueError("No Venice API key provided.")

    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }

    # Try the live models endpoint
    try:
        r = requests.get(
            "%s/models" % API_ROOT,
            headers=headers,
            timeout=30,
        )
        if r.status_code == 200:
            # Safely parse — don't trust content-type
            try:
                text = r.text.strip()
                data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                data = None

            if isinstance(data, dict):
                models = data.get("data") or []
                if isinstance(models, list) and len(models) > 0:
                    out = []
                    for m in models:
                        if not isinstance(m, dict):
                            continue
                        mid = m.get("id", "")
                        if mid:
                            out.append({
                                "id": mid,
                                "display_name": m.get("name", mid),
                            })
                    if out:
                        out.sort(key=lambda x: x["id"])
                        return out
    except requests.RequestException:
        pass  # Network error — fall through to fallback

    # Fallback: return models from Venice documentation
    out = [{"id": m["id"], "display_name": m["display_name"]} for m in VENICE_MODELS]
    out.sort(key=lambda x: x["id"])
    return out


def _err(resp):
    try:
        text = resp.text.strip()
        data = json.loads(text)
        if isinstance(data, dict):
            e = data.get("error", {})
            if isinstance(e, dict):
                msg = e.get("message", resp.text[:200])
            elif isinstance(e, str):
                msg = e
            else:
                msg = str(e)
        else:
            msg = resp.text[:200]
        return "Venice API %s: %s" % (resp.status_code, msg)
    except Exception:
        return "Venice API %s: %s" % (resp.status_code, resp.text[:200])


def generate(api_key, model_id, idea, pil_image=None, density="normal"):
    """Call Venice chat completions and return a normalized caption dict."""
    if not api_key:
        raise ValueError("No Venice API key provided.")
    if not model_id:
        raise ValueError("No Venice model selected.")

    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }

    user_content = []
    user_content.append({
        "type": "text",
        "text": build_user_prompt(idea, pil_image is not None),
    })

    if pil_image is not None:
        buf = _io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,%s" % b64,
            },
        })

    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": get_system_prompt(density)},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "venice_parameters": {
            "include_venice_system_prompt": False,
        },
    }

    url = "%s/chat/completions" % API_ROOT
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise ValueError(_err(r))

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Venice returned no choices.")

    text = choices[0].get("message", {}).get("content", "")
    if not text:
        raise ValueError("Venice returned empty content.")

    return parse_caption(text)