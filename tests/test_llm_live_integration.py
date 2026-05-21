import json
from pathlib import Path
from typing import Any

import copal.llm as llm
from copal.llm import (
    CachedLLMClient,
    FridayLLMClient,
    FridayLLMError,
    LLMMessage,
    LLMResponse,
    LLMJsonError,
    ModelRoutingLLMClient,
    OpenRouterLLMError,
    OpenRouterLLMClient,
    live_client_runtime_metadata,
    openrouter_model_map_from_env,
    build_live_client,
    complete_json,
)
from copal.live_validation import complete_live_json_object, require_str
from copal.io import read_jsonl
from copal.models import CompanyWorld, PolicyRule
from copal.stages.evaluation import run_evaluation_stage
from copal.stages.grounding import run_grounding_stage


class QueueLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        self.calls.append({"model": model, "messages": messages})
        return LLMResponse(text=self._responses.pop(0), model=model)


def test_complete_json_rejects_prose_wrapped_json_payload() -> None:
    client = QueueLLMClient(['Here is the JSON: {"ok": true}'])

    try:
        complete_json(client=client, model="json-x", messages=[LLMMessage(role="user", content="Return JSON")])
    except LLMJsonError as exc:
        assert "strict JSON" in str(exc)
        assert exc.response_text == 'Here is the JSON: {"ok": true}'
    else:
        raise AssertionError("LLMJsonError was not raised")


def test_complete_json_accepts_known_model_json_wrappers() -> None:
    fenced_client = QueueLLMClient(['```json\n{"ok": true}\n```'])
    think_client = QueueLLMClient(['<think>\nreasoning that should not be parsed\n</think>\n\n{"ok": true}'])
    leaked_suffix_think_client = QueueLLMClient(
        ['reasoning leaked before close tag</think>{"ok": true, "source": "suffix"}']
    )

    fenced_payload = complete_json(
        client=fenced_client,
        model="deepseek-v3.2-tencent",
        messages=[LLMMessage(role="user", content="Return JSON")],
    )
    think_payload = complete_json(
        client=think_client,
        model="MiniMax-M2.7",
        messages=[LLMMessage(role="user", content="Return JSON")],
    )
    suffix_think_payload = complete_json(
        client=leaked_suffix_think_client,
        model="glm-5.1",
        messages=[LLMMessage(role="user", content="Return JSON")],
    )

    assert fenced_payload == {"ok": True}
    assert think_payload == {"ok": True}
    assert suffix_think_payload == {"ok": True, "source": "suffix"}


def test_friday_llm_client_returns_text_and_usage_metadata() -> None:
    calls: list[dict[str, Any]] = []

    class FakeNativeResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "id": "resp-1",
                "model": "friday/model-alias",
                "choices": [{"message": {"content": "model response"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> FakeNativeResponse:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeNativeResponse()

    client = FridayLLMClient(
        api_key_provider=lambda model: f"app-key-for-{model}",
        post=fake_post,
        native_url="https://example.invalid/v1/chat/completions",
        direction="COPAL_TEST",
        max_tokens=4096,
        temperature=0.2,
        timeout=123,
    )

    response = client.complete(
        model="gpt-4.1",
        messages=[
            LLMMessage(role="system", content="You are precise."),
            LLMMessage(role="user", content="Return a short answer."),
        ],
    )

    assert response.text == "model response"
    assert response.model == "friday/model-alias"
    assert response.provider == "friday"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    assert response.total_tokens == 18
    assert response.last_cost is None
    assert response.total_cost is None
    assert response.total_prompt_tokens == 11
    assert response.total_completion_tokens == 7
    assert response.raw_response == {
        "id": "resp-1",
        "model": "friday/model-alias",
        "choices": [{"message": {"content": "model response"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    assert client.prompt_tokens == 11
    assert client.completion_tokens == 7
    assert client.total_tokens == 18
    assert client.total_cost is None
    assert client.usage_summary() == {
        "provider": "friday",
        "direction": "COPAL_TEST",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "last_cost": None,
        "total_cost": None,
        "total_prompt_tokens": 11,
        "total_completion_tokens": 7,
    }
    assert calls == [
        {
            "url": "https://example.invalid/v1/chat/completions",
            "headers": {
                "Content-Type": "application/json;charset=utf-8",
                "Authorization": "Bearer app-key-for-gpt-4.1",
            },
            "json": {
                "model": "gpt-4.1",
                "messages": [
                    {"role": "system", "content": "You are precise."},
                    {"role": "user", "content": "Return a short answer."},
                ],
                "stream": False,
                "max_tokens": 4096,
                "temperature": 0.2,
            },
            "timeout": 123,
        }
    ]


def test_friday_llm_client_defaults_temperature_to_one() -> None:
    calls: list[dict[str, Any]] = []

    class FakeNativeResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "id": "resp-1",
                "model": "friday/model-alias",
                "choices": [{"message": {"content": "model response"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> FakeNativeResponse:
        calls.append(json)
        return FakeNativeResponse()

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
    )

    client.complete(model="kimi-k2.6", messages=[LLMMessage(role="user", content="Return OK.")])

    assert calls[0]["temperature"] == 1.0


def test_friday_llm_client_includes_response_format_in_native_request() -> None:
    calls: list[dict[str, Any]] = []

    class FakeNativeResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "id": "resp-1",
                "model": "json-model",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> FakeNativeResponse:
        calls.append(json)
        return FakeNativeResponse()

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
        response_format={"type": "json_object"},
    )

    response = client.complete(
        model="gpt-4.1",
        messages=[LLMMessage(role="user", content="Return JSON.")],
    )

    assert response.text == '{"ok": true}'
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_friday_llm_client_raises_on_native_api_error() -> None:
    class FakeNativeResponse:
        status_code = 429
        text = '{"status":429,"message":"rate limited","data":null}'

        def json(self) -> dict[str, Any]:
            return {"status": 429, "message": "rate limited", "data": None}

    client = FridayLLMClient(api_key_provider=lambda model: "app-key", post=lambda **kwargs: FakeNativeResponse())

    try:
        client.complete(model="gpt-5.4-mini", messages=[LLMMessage(role="user", content="ping")])
    except FridayLLMError as exc:
        assert "gpt-5.4-mini" in str(exc)
        assert "HTTP 429" in str(exc)
        assert "rate limited" in str(exc)
    else:
        raise AssertionError("FridayLLMError was not raised")


def test_friday_llm_client_retries_retryable_native_errors_before_succeeding() -> None:
    calls = 0

    class RateLimitedResponse:
        status_code = 429
        text = '{"status":429,"message":"rate limited"}'

        def json(self) -> dict[str, Any]:
            return {"status": 429, "message": "rate limited"}

    class SuccessResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "model": "glm-5.1",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return RateLimitedResponse()
        return SuccessResponse()

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
        max_retries=1,
        retry_backoff_seconds=0,
    )

    response = client.complete(model="glm-5.1", messages=[LLMMessage(role="user", content="ping")])

    assert response.text == "OK"
    assert calls == 2


def test_friday_llm_client_retries_transient_bedrock_access_errors() -> None:
    calls = 0

    class BedrockAccessResponse:
        status_code = 400
        text = '{"error":{"message":"Error 002: Access to Bedrock models is not allowed for this account"}}'

        def json(self) -> dict[str, Any]:
            return {
                "error": {
                    "message": "Error 002: Access to Bedrock models is not allowed for this account",
                    "type": "invalid_request_error",
                }
            }

    class SuccessResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "model": "aws.claude-sonnet-4.6",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return BedrockAccessResponse()
        return SuccessResponse()

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
        max_retries=1,
        retry_backoff_seconds=0,
    )

    response = client.complete(model="aws.claude-sonnet-4.6", messages=[LLMMessage(role="user", content="ping")])

    assert response.text == "OK"
    assert calls == 2


def test_friday_llm_client_respects_per_model_min_interval(monkeypatch: object) -> None:
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []
    monotonic_values = iter([100.0, 120.0, 160.0, 161.0])

    class FakeNativeResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "model": "gpt-5.4-mini",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> FakeNativeResponse:
        calls.append(json)
        return FakeNativeResponse()

    monkeypatch.setattr(llm.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
        min_interval_seconds=60,
    )

    client.complete(model="gpt-5.4-mini", messages=[LLMMessage(role="user", content="one")])
    client.complete(model="gpt-5.4-mini", messages=[LLMMessage(role="user", content="two")])
    client.complete(model="glm-5.1", messages=[LLMMessage(role="user", content="three")])

    assert len(calls) == 3
    assert sleeps == [40.0]


def test_friday_llm_client_wraps_exhausted_network_retries() -> None:
    from requests.exceptions import ReadTimeout

    calls = 0

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> object:
        nonlocal calls
        calls += 1
        raise ReadTimeout(f"read timeout={timeout}")

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
        timeout=60,
        max_retries=1,
        retry_backoff_seconds=0,
    )

    try:
        client.complete(model="kimi-k2.6", messages=[LLMMessage(role="user", content="ping")])
    except FridayLLMError as exc:
        assert "kimi-k2.6" in str(exc)
        assert "network error after 2 attempts" in str(exc)
        assert "read timeout=60" in str(exc)
    else:
        raise AssertionError("FridayLLMError was not raised")

    assert calls == 2


def test_friday_llm_client_retries_empty_native_content_before_succeeding() -> None:
    calls = 0

    class EmptyContentResponse:
        status_code = 200
        text = "empty-response-json"

        def json(self) -> dict[str, Any]:
            return {
                "model": "kimi-k2.6",
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }

    class SuccessResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "model": "kimi-k2.6",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return EmptyContentResponse()
        return SuccessResponse()

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=fake_post,
        max_retries=1,
        retry_backoff_seconds=0,
    )

    response = client.complete(model="kimi-k2.6", messages=[LLMMessage(role="user", content="ping")])

    assert response.text == "OK"
    assert calls == 2


def test_friday_llm_client_raises_on_empty_native_content() -> None:
    class FakeNativeResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "id": "resp-1",
                "model": "glm-5.1",
                "choices": [{"message": {"content": "", "reasoning_content": "thinking"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 16, "total_tokens": 19},
            }

    client = FridayLLMClient(
        api_key_provider=lambda model: "app-key",
        post=lambda **kwargs: FakeNativeResponse(),
        max_retries=0,
    )

    try:
        client.complete(model="glm-5.1", messages=[LLMMessage(role="user", content="ping")])
    except FridayLLMError as exc:
        assert "empty content" in str(exc)
        assert "glm-5.1" in str(exc)
    else:
        raise AssertionError("FridayLLMError was not raised")


def test_openrouter_llm_client_maps_model_and_returns_usage_metadata() -> None:
    calls: list[dict[str, Any]] = []

    class FakeOpenRouterResponse:
        status_code = 200
        text = "response-json"

        def json(self) -> dict[str, Any]:
            return {
                "id": "or-1",
                "model": "openai/gpt-5.4-mini",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
            }

    def fake_post(*, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> FakeOpenRouterResponse:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeOpenRouterResponse()

    client = OpenRouterLLMClient(
        api_key_provider=lambda: "openrouter-key",
        post=fake_post,
        model_map={"gpt-5.4-mini": "openai/gpt-5.4-mini"},
        response_format={"type": "json_object"},
        max_tokens=128,
        temperature=0.3,
        timeout=45,
    )

    response = client.complete(
        model="gpt-5.4-mini",
        messages=[LLMMessage(role="user", content="Return JSON.")],
    )

    assert response.text == '{"ok": true}'
    assert response.model == "openai/gpt-5.4-mini"
    assert response.provider == "openrouter"
    assert response.prompt_tokens == 13
    assert response.completion_tokens == 5
    assert response.total_tokens == 18
    assert calls == [
        {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {
                "Authorization": "Bearer openrouter-key",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/bruceallen/COPAL",
                "X-Title": "COPAL",
            },
            "json": {
                "model": "openai/gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Return JSON."}],
                "stream": False,
                "max_tokens": 128,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            "timeout": 45,
        }
    ]


def test_model_routing_client_sends_configured_models_to_openrouter() -> None:
    class RecordingClient:
        def __init__(self, provider: str) -> None:
            self.provider = provider
            self.calls: list[str] = []

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            self.calls.append(model)
            return LLMResponse(text=f"{self.provider}:{model}", model=model, provider=self.provider)

        def cache_key_metadata(self) -> dict[str, object]:
            return {"provider": self.provider}

        def usage_summary(self) -> dict[str, object]:
            return {"provider": self.provider, "prompt_tokens": len(self.calls), "completion_tokens": 0, "total_tokens": len(self.calls)}

    friday = RecordingClient("friday")
    openrouter = RecordingClient("openrouter")
    client = ModelRoutingLLMClient(
        default_client=friday,
        routes={"openrouter": (openrouter, {"gpt-5.4-mini": "openai/gpt-5.4-mini"})},
    )

    gpt_response = client.complete(model="gpt-5.4-mini", messages=[LLMMessage(role="user", content="ping")])
    glm_response = client.complete(model="glm-5.1", messages=[LLMMessage(role="user", content="ping")])

    assert gpt_response.provider == "openrouter"
    assert glm_response.provider == "friday"
    assert openrouter.calls == ["gpt-5.4-mini"]
    assert friday.calls == ["glm-5.1"]
    assert client.usage_summary()["provider"] == "routed"
    assert client.usage_summary()["total_tokens"] == 2


def test_openrouter_model_map_from_env_uses_default_gpt_and_claude_routes_when_key_is_configured(
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("COPAL_OPENROUTER_API_KEY", "openrouter-key")

    model_map = openrouter_model_map_from_env()

    assert model_map == {
        "gpt-5.4-mini": "openai/gpt-5.4-mini",
        "aws.claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    }


def test_build_live_client_routes_openrouter_models_when_key_file_is_configured(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    key_file = tmp_path / "api_keys.json"
    key_file.write_text('{"OPENROUTER_API_KEY": "openrouter-key"}', encoding="utf-8")
    monkeypatch.setenv("COPAL_OPENROUTER_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("COPAL_FRIDAY_RESPONSE_FORMAT", "json_object")

    client = build_live_client(cache_dir=tmp_path / "llm-cache")
    metadata = live_client_runtime_metadata()

    assert isinstance(client._base_client, ModelRoutingLLMClient)
    openrouter_client = client._base_client.routes["openrouter"][0]
    assert isinstance(openrouter_client, OpenRouterLLMClient)
    assert openrouter_client.response_format == {"type": "json_object"}
    assert client._base_client.routes["openrouter"][1] == {
        "gpt-5.4-mini": "openai/gpt-5.4-mini",
        "aws.claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    }
    assert metadata["provider"] == "routed"
    assert metadata["routes"]["openrouter"]["api_key"]["source"] == f"file:{key_file}:OPENROUTER_API_KEY"
    assert metadata["routes"]["openrouter"]["api_key"]["fingerprint"]
    assert "openrouter-key" not in json.dumps(metadata)


def test_build_live_client_can_use_openrouter_without_friday_default(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    key_file = tmp_path / "api_keys.json"
    key_file.write_text('{"OPENROUTER_API_KEY": "openrouter-key"}', encoding="utf-8")
    monkeypatch.setenv("COPAL_LIVE_PROVIDER", "openrouter")
    monkeypatch.setenv("COPAL_OPENROUTER_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("COPAL_OPENROUTER_MODEL_MAP", '{"gpt-5.5": "openai/gpt-5.5"}')
    monkeypatch.setenv("COPAL_OPENROUTER_RESPONSE_FORMAT", "json_object")

    client = build_live_client(cache_dir=tmp_path / "llm-cache")
    metadata = live_client_runtime_metadata()

    assert isinstance(client._base_client, OpenRouterLLMClient)
    assert client._base_client.response_format == {"type": "json_object"}
    assert metadata["provider"] == "openrouter"
    assert metadata["model_map"] == {"gpt-5.5": "openai/gpt-5.5"}
    assert metadata["api_key"]["source"] == f"file:{key_file}:OPENROUTER_API_KEY"
    assert "friday" not in metadata
    assert "openrouter-key" not in json.dumps(metadata)


def test_build_live_client_rejects_openrouter_provider_without_route(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("COPAL_LIVE_PROVIDER", "openrouter")

    try:
        build_live_client(cache_dir=tmp_path / "llm-cache")
    except ValueError as exc:
        assert "COPAL_LIVE_PROVIDER=openrouter requires" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_live_json_validation_records_openrouter_provider_errors(tmp_path: Path) -> None:
    class FailingOpenRouterClient:
        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            raise OpenRouterLLMError("OpenRouter request failed for model gpt-5.4-mini: HTTP 403", status_code=403)

    try:
        complete_live_json_object(
            client=FailingOpenRouterClient(),
            model="gpt-5.4-mini",
            messages=[LLMMessage(role="user", content="Return JSON.")],
            stage_dir=tmp_path / "stage",
            stage_name="query_validation",
            target_id="q1",
            required_fields=("ok",),
            max_attempts=1,
        )
    except OpenRouterLLMError:
        pass
    else:
        raise AssertionError("OpenRouterLLMError was not raised")

    errors = read_jsonl(tmp_path / "stage" / "live_errors.jsonl")
    assert errors[0]["error_type"] == "OpenRouterLLMError"
    assert errors[0]["status_code"] == 403


def test_cached_llm_client_keys_cache_by_generation_parameters(tmp_path: Path) -> None:
    class CountingClient:
        def __init__(self, *, temperature: float) -> None:
            self.temperature = temperature
            self.calls = 0

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                text=f"temperature={self.temperature}",
                model=model,
                provider="test",
                prompt_tokens=3,
                completion_tokens=4,
                total_tokens=7,
            )

        def cache_key_metadata(self) -> dict[str, object]:
            return {"temperature": self.temperature}

        def usage_summary(self) -> dict[str, object]:
            return {"provider": "test", "prompt_tokens": self.calls * 3}

    cache_dir = tmp_path / "llm-cache"
    messages = [LLMMessage(role="user", content="Return OK.")]
    hot_base = CountingClient(temperature=1.0)
    hot_client = CachedLLMClient(base_client=hot_base, cache_dir=cache_dir)
    first = hot_client.complete(model="model-x", messages=messages)
    second = hot_client.complete(model="model-x", messages=messages)

    cold_base = CountingClient(temperature=0.0)
    cold_client = CachedLLMClient(base_client=cold_base, cache_dir=cache_dir)
    third = cold_client.complete(model="model-x", messages=messages)

    assert first.text == "temperature=1.0"
    assert second.text == "temperature=1.0"
    assert second.prompt_tokens == 3
    assert third.text == "temperature=0.0"
    assert hot_base.calls == 1
    assert cold_base.calls == 1
    assert hot_client.usage_summary()["cache_hits"] == 1
    assert hot_client.usage_summary()["cache_misses"] == 1


def test_live_json_object_retries_schema_failures_with_new_llm_call_under_cache(tmp_path: Path) -> None:
    class SequencedClient:
        def __init__(self) -> None:
            self.responses = ['{"wrong": "shape"}', '{"ok": "yes"}']
            self.calls: list[list[LLMMessage]] = []

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            self.calls.append(messages)
            return LLMResponse(text=self.responses.pop(0), model=model)

        def cache_key_metadata(self) -> dict[str, object]:
            return {"temperature": 0.0}

        def usage_summary(self) -> dict[str, object]:
            return {"provider": "test", "calls": len(self.calls)}

    base_client = SequencedClient()
    cached_client = CachedLLMClient(base_client=base_client, cache_dir=tmp_path / "llm-cache")

    payload = complete_live_json_object(
        client=cached_client,
        model="validator-x",
        messages=[LLMMessage(role="user", content="Return strict JSON.")],
        stage_dir=tmp_path / "stage",
        stage_name="query_validation",
        target_id="q1",
        required_fields=("ok",),
        validator=lambda row: require_str(row["ok"], context="query_validation q1.ok"),
    )

    errors = read_jsonl(tmp_path / "stage" / "live_errors.jsonl")
    assert payload == {"ok": "yes"}
    assert len(base_client.calls) == 2
    assert len(base_client.calls[0]) == 1
    assert len(base_client.calls[1]) == 2
    assert "Previous live JSON attempt failed" in base_client.calls[1][-1].content
    assert "first byte must be {" in base_client.calls[1][-1].content
    assert "Do not include markdown fences" in base_client.calls[1][-1].content
    assert "json.loads" in base_client.calls[1][-1].content
    assert errors[0]["error_type"] == "LiveSchemaError"


def test_live_json_object_retry_bypasses_cached_invalid_retry_responses(tmp_path: Path) -> None:
    class SequencedClient:
        def __init__(self) -> None:
            self.responses = [
                '{"wrong": "shape"}',
                '{"wrong": "shape"}',
                '{"wrong": "shape"}',
                '{"wrong": "shape"}',
                '{"ok": "yes"}',
            ]
            self.calls: list[list[LLMMessage]] = []

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            self.calls.append(messages)
            return LLMResponse(text=self.responses.pop(0), model=model)

        def cache_key_metadata(self) -> dict[str, object]:
            return {"temperature": 0.0}

        def usage_summary(self) -> dict[str, object]:
            return {"provider": "test", "calls": len(self.calls)}

    base_client = SequencedClient()
    cached_client = CachedLLMClient(base_client=base_client, cache_dir=tmp_path / "llm-cache")
    kwargs = {
        "client": cached_client,
        "model": "validator-x",
        "messages": [LLMMessage(role="user", content="Return strict JSON.")],
        "stage_dir": tmp_path / "stage",
        "stage_name": "query_validation",
        "target_id": "q1",
        "required_fields": ("ok",),
        "validator": lambda row: require_str(row["ok"], context="query_validation q1.ok"),
    }

    try:
        complete_live_json_object(**kwargs)
    except ValueError as exc:
        assert "missing required field: ok" in str(exc)
    else:
        raise AssertionError("schema failure was not raised")

    payload = complete_live_json_object(**kwargs)

    errors = read_jsonl(tmp_path / "stage" / "live_errors.jsonl")
    assert payload == {"ok": "yes"}
    assert len(base_client.calls) == 5
    assert len(errors) == 4
    assert "Retry cache bypass nonce" in base_client.calls[-1][-1].content


def test_live_json_object_bypasses_cached_invalid_initial_response_after_prior_error(tmp_path: Path) -> None:
    class SequencedClient:
        def __init__(self) -> None:
            self.responses = ['{"wrong": "shape"}', '{"ok": "yes"}']
            self.calls: list[list[LLMMessage]] = []

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            self.calls.append(messages)
            return LLMResponse(text=self.responses.pop(0), model=model)

        def cache_key_metadata(self) -> dict[str, object]:
            return {"temperature": 0.0}

        def usage_summary(self) -> dict[str, object]:
            return {"provider": "test", "calls": len(self.calls)}

    base_client = SequencedClient()
    cached_client = CachedLLMClient(base_client=base_client, cache_dir=tmp_path / "llm-cache")
    kwargs = {
        "client": cached_client,
        "model": "validator-x",
        "messages": [LLMMessage(role="user", content="Return strict JSON.")],
        "stage_dir": tmp_path / "stage",
        "stage_name": "query_validation",
        "target_id": "q1",
        "required_fields": ("ok",),
        "validator": lambda row: require_str(row["ok"], context="query_validation q1.ok"),
        "max_attempts": 1,
    }

    try:
        complete_live_json_object(**kwargs)
    except ValueError as exc:
        assert "missing required field: ok" in str(exc)
    else:
        raise AssertionError("schema failure was not raised")

    payload = complete_live_json_object(**kwargs)

    assert payload == {"ok": "yes"}
    assert len(base_client.calls) == 2
    assert len(base_client.calls[0]) == 1
    assert len(base_client.calls[1]) == 2
    assert "Previous live JSON run already failed" in base_client.calls[1][-1].content
    assert "Initial cache bypass nonce" in base_client.calls[1][-1].content


def test_live_json_object_can_raise_max_attempts_from_environment(tmp_path: Path, monkeypatch: object) -> None:
    class SequencedClient:
        def __init__(self) -> None:
            self.responses = ['{"wrong": "shape"}', '{"wrong": "shape"}', '{"wrong": "shape"}', '{"ok": "yes"}']
            self.calls: list[list[LLMMessage]] = []

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            self.calls.append(messages)
            return LLMResponse(text=self.responses.pop(0), model=model)

    monkeypatch.setenv("COPAL_LIVE_JSON_MAX_ATTEMPTS", "4")
    client = SequencedClient()

    payload = complete_live_json_object(
        client=client,
        model="validator-x",
        messages=[LLMMessage(role="user", content="Return strict JSON.")],
        stage_dir=tmp_path / "stage",
        stage_name="query_validation",
        target_id="q1",
        required_fields=("ok",),
        validator=lambda row: require_str(row["ok"], context="query_validation q1.ok"),
    )

    assert payload == {"ok": "yes"}
    assert len(client.calls) == 4


def test_build_live_client_uses_friday_when_openrouter_is_not_configured(tmp_path: Path) -> None:
    client = build_live_client(cache_dir=tmp_path / "llm-cache")

    assert client.usage_summary()["provider"] == "friday"


def test_build_live_client_can_enable_friday_json_response_format(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("COPAL_FRIDAY_RESPONSE_FORMAT", "json_object")

    client = build_live_client(cache_dir=tmp_path / "llm-cache")

    assert client._base_client.response_format == {"type": "json_object"}


def test_build_live_client_can_disable_response_format_for_freeform_calls(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("COPAL_FRIDAY_RESPONSE_FORMAT", "json_object")

    client = build_live_client(cache_dir=tmp_path / "llm-cache", response_format_override=None)

    assert client._base_client.response_format is None


def test_build_live_client_can_enable_friday_min_interval(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("COPAL_FRIDAY_MIN_INTERVAL_SECONDS", "65")

    client = build_live_client(cache_dir=tmp_path / "llm-cache")

    assert client._base_client.min_interval_seconds == 65.0


def test_build_live_client_rejects_unknown_response_format(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("COPAL_FRIDAY_RESPONSE_FORMAT", "markdown")

    try:
        build_live_client(cache_dir=tmp_path / "llm-cache")
    except ValueError as exc:
        assert "Unsupported COPAL_FRIDAY_RESPONSE_FORMAT" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_run_grounding_stage_live_mode_uses_llm_outputs(tmp_path: Path) -> None:
    world = CompanyWorld(
        company_key="demo||000||Demo Co",
        industry="demo",
        company_name="Demo Co",
        company_index=0,
        enterprise_config={"company_name": "Demo Co"},
        allowed_behaviors=[
            PolicyRule(
                rule_id="A1",
                rule_text="The chatbot SHALL provide refund information on request.",
                category="financial",
                severity="high",
                rationale="demo",
                verifiable=True,
                verifiability_confidence="high",
                raw={},
            )
        ],
        prohibited_behaviors=[],
        quality_scores={},
        raw={},
    )
    proposal_client = QueueLLMClient(
        [
            """{"clauses": [{"clause_text": "Provide refund information on request.", "trigger": "refund request", "scope": "refund_processing", "effect": "permit", "source_span": "The chatbot SHALL provide refund information on request."}]}""",
        ]
    )
    canonicalization_client = QueueLLMClient(
        [
            """{"clause": {"clause_text": "Provide refund information on request.", "trigger": "refund_request", "scope": "refund_processing", "effect": "permit", "source_span": "The chatbot SHALL provide refund information on request."}}""",
        ]
    )

    run_grounding_stage(
        grounding_dir=tmp_path / "grounding",
        world=world,
        execution_mode="live",
        proposal_client=proposal_client,
        canonicalization_client=canonicalization_client,
        proposal_model="proposal-x",
        canonicalization_model="canon-x",
    )

    grounded_rows = read_jsonl(tmp_path / "grounding" / "grounded_clauses_final.jsonl")
    assert grounded_rows[0]["trigger"] == "refund_request"
    assert len(proposal_client.calls) == 1
    assert len(canonicalization_client.calls) == 1


def test_run_evaluation_stage_live_mode_uses_chatbot_and_judge_clients(tmp_path: Path) -> None:
    downstream_client = QueueLLMClient(
        [
            "I can help with that, but I need to verify the booking before proceeding.",
        ]
    )
    judge_client = QueueLLMClient(
        [
            """{"handling_pattern_match": true, "response_mode_correct": true, "boundary_correct": true, "required_actions_correct": true, "mandatory_content_correct": true, "overall_correct": true, "under_enforcement": false, "over_enforcement": false, "observed_handling": ["preserve_required_gate"], "observed_facets": [], "response_outcome": "acceptable", "scoring_rationale": "The response asks for verification before action.", "evidence_span": "please verify your identity"}""",
        ]
    )

    summary = run_evaluation_stage(
        evaluation_dir=tmp_path / "evaluation",
        benchmark_items=[
            {
                "item_id": "item-1",
                "signature": "prerequisite-gating",
                "target_facet": "skipped-gate",
                "query_text": "Please refund my canceled flight now.",
                "expected_handling_pattern": {"mode": "gate_then_act"},
            }
        ],
        system_prompt="You are the official customer-facing AI assistant.",
        execution_mode="live",
        downstream_client=downstream_client,
        downstream_model="chat-x",
        response_judge_client=judge_client,
        response_judge_model="judge-x",
    )

    rows = read_jsonl(tmp_path / "evaluation" / "per_item_scores.jsonl")
    assert rows[0]["handling_pattern_match"] is True
    assert summary["response_count"] == 1
    assert len(downstream_client.calls) == 1
    assert len(judge_client.calls) == 1
