# LM interfaces

Model selection is driven by the `protocol` field in the model config —
`model_name` is the serving name, sent to the backend verbatim. All three
interfaces speak the chat-completions message format toward the agents and
translate to their wire protocol natively (no litellm).

* `protocol: chat` (default) → `openai_chat.py` — OpenAI chat completions
   (legacy `/v1/chat/completions`), via the official `openai` SDK.
* `protocol: anthropic` → `anthropic.py` — Anthropic Messages API
   (`/v1/messages`), via the official `anthropic` SDK. Handles cache_control
   breakpoints, thinking-block replay, and images in tool results.
* `protocol: responses` → `openai_responses.py` — OpenAI
   [Responses API](https://developers.openai.com/api/reference/responses/overview)
   (`/v1/responses`).
* `test_models.py` - Deterministic models that can be used for internal testing

Blackbox scaffolds never query through the model object (it only carries the
gateway config for the harness), so their configs simply omit `protocol`.

Config example:

```yaml
model:
  model_name: "gpt-5"
  protocol: "chat"
  model_kwargs:
    base_url: "${OPENAI_BASE_URL:-https://api.openai.com/v1}"
    api_key: "${OPENAI_API_KEY}"
```

String values may reference environment variables as `${VAR}` or
`${VAR:-default}`; they are expanded when the config is loaded.

All three accept multimodal content: message `content` may be a list of
`{"type": "text", ...}` / `{"type": "image_url", "image_url": {"url": ...}}`
parts (http(s) or data URLs), which each interface converts to its wire format.
