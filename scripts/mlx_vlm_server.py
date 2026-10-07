#!/usr/bin/env python3
"""Run upstream mlx_vlm with core request defaults, without package edits."""

from __future__ import annotations
import json
import sys


class PolicyMiddleware:
    """Fill omitted sampling fields before upstream Pydantic normalization.

    Explicit values (including zero and null) remain upstream's responsibility.
    Images, tools, messages and streaming response events pass through intact.
    """

    PATHS = {
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/responses",
        "/responses",
    }
    KEYS = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "repetition_context_size",
        "presence_context_size",
        "enable_thinking",
        "reasoning_effort",
        "thinking_budget",
    }

    def __init__(self, app, policy, thinking=None):
        self.app, self.policy, self.thinking = app, policy, thinking

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self.PATHS
        ):
            return await self.app(scope, receive, send)
        events = []
        while True:
            event = await receive()
            events.append(event)
            if event["type"] != "http.request" or not event.get("more_body", False):
                break
        if events[-1]["type"] == "http.request":
            body = b"".join(event.get("body", b"") for event in events)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                payload = None
            if isinstance(payload, dict):
                defaults = dict(self.policy["sampling"])
                thinking = payload.get("enable_thinking", self.thinking)
                nested = payload.get("chat_template_kwargs", {})
                if isinstance(nested, dict):
                    thinking = nested.get("enable_thinking", thinking)
                if thinking is True:
                    defaults.update(
                        self.policy.get("sampling_recipes", {})
                        .get("thinking", {})
                        .get("sampling", {})
                    )
                if self.thinking is not None:
                    defaults["enable_thinking"] = self.thinking
                if "repetition_context_size" in defaults:
                    defaults["presence_context_size"] = defaults[
                        "repetition_context_size"
                    ]
                # Model policy reasoning_budget maps to upstream VLM's field.
                if "reasoning_budget" in defaults:
                    defaults["thinking_budget"] = defaults["reasoning_budget"]
                if isinstance(nested, dict) and "enable_thinking" in nested:
                    defaults["enable_thinking"] = nested["enable_thinking"]
                for key in self.KEYS:
                    if key in defaults:
                        payload.setdefault(key, defaults[key])
                body = json.dumps(payload).encode()
                events = [{"type": "http.request", "body": body, "more_body": False}]
                scope = dict(scope)
                scope["headers"] = [
                    (key, value)
                    for key, value in scope.get("headers", [])
                    if key.lower() != b"content-length"
                ] + [(b"content-length", str(len(body)).encode())]
        pending = iter(events)

        async def replay():
            try:
                return next(pending)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)


def main():
    from model_policy import resolve_policy
    from mlx_vlm.server import cli
    import mlx_vlm.server as server
    import uvicorn

    index = sys.argv.index("--model")
    policy = resolve_policy(sys.argv[index + 1], "mlx")
    thinking = True if "--enable-thinking" in sys.argv else None
    original = uvicorn.run

    def run(target, *args, **kwargs):
        server.app = PolicyMiddleware(server.app, policy, thinking)
        return original(target, *args, **kwargs)

    uvicorn.run = run
    cli.main()


if __name__ == "__main__":
    main()
