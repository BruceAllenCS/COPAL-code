from __future__ import annotations

import json
import os
import time
from threading import Lock
from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, Callable, Protocol

from copal.io import ensure_directory, read_json, write_json

FRIDAY_NATIVE_CHAT_URL = os.getenv("COPAL_FRIDAY_NATIVE_CHAT_URL", os.getenv("COPAL_OPENAI_COMPATIBLE_CHAT_URL", ""))
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL_MAP = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "aws.claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
}
LIVE_PROVIDER_MODES = ("auto", "friday", "openrouter", "routed")
_USE_ENV_RESPONSE_FORMAT = object()


@dataclass(slots=True)
class LLMMessage:
    role: str
    content: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    provider: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_cost: float | None = None
    total_cost: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    raw_response: Any | None = None


class LLMClient(Protocol):
    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        """Return a single text completion for the given messages."""


class LLMProviderError(RuntimeError):
    """Raised when a live model provider does not return a usable completion."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FridayLLMError(LLMProviderError):
    """Raised when the Friday native API does not return a usable completion."""


class OpenRouterLLMError(LLMProviderError):
    """Raised when the OpenRouter API does not return a usable completion."""


class LLMJsonError(ValueError):
    """Raised when a model response is not exactly the requested JSON payload."""

    def __init__(self, message: str, *, response_text: str) -> None:
        super().__init__(message)
        self.response_text = response_text


def _record_response_usage(client: Any, response: LLMResponse) -> None:
    with client._usage_lock:
        client.prompt_tokens += response.prompt_tokens
        client.completion_tokens += response.completion_tokens
        client.total_tokens += response.total_tokens
        client.last_cost = response.last_cost
        client.total_cost = response.total_cost
        client.total_prompt_tokens = client.prompt_tokens
        client.total_completion_tokens = client.completion_tokens
        client.last_response = response.raw_response


class FridayLLMClient:
    """Friday native OpenAI-compatible client with local usage metadata capture."""

    def __init__(
        self,
        *,
        direction: str = "COPAL",
        max_tokens: int = 8000,
        temperature: float = 1.0,
        timeout: int = 240,
        response_format: dict[str, Any] | None = None,
        use_vision: bool = False,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        min_interval_seconds: float = 0.0,
        api_key_provider: Callable[[str], str] | None = None,
        post: Callable[..., Any] | None = None,
        native_url: str = FRIDAY_NATIVE_CHAT_URL,
    ) -> None:
        if use_vision:
            raise ValueError("Friday native vision execution is not implemented in COPAL")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self.direction = direction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.response_format = response_format
        self.native_url = native_url
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.min_interval_seconds = min_interval_seconds
        self._api_key_provider = api_key_provider or self._load_api_key_provider()
        self._post = post or self._load_post()
        self._retryable_network_errors = self._load_retryable_network_errors()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.last_cost: float | None = None
        self.total_cost: float | None = None
        self.total_prompt_tokens: int | None = None
        self.total_completion_tokens: int | None = None
        self.last_response: Any | None = None
        self._usage_lock = Lock()
        self._rate_limit_lock = Lock()
        self._last_request_at_by_model: dict[str, float] = {}

    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [asdict(message) for message in messages],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.response_format is not None:
            payload["response_format"] = self.response_format

        attempt = 0
        while True:
            try:
                self._respect_min_interval(model)
                native_response = self._post(
                    url=self.native_url,
                    headers={
                        "Content-Type": "application/json;charset=utf-8",
                        "Authorization": f"Bearer {self._api_key_provider(model)}",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                body = self._native_body(native_response=native_response, model=model)
                response = self._response_from_native_body(body=body, requested_model=model)
                self._record_usage(response)
                return response
            except FridayLLMError as exc:
                if not self._is_retryable_friday_error(exc) or attempt >= self.max_retries:
                    raise
            except self._retryable_network_errors as exc:
                if attempt >= self.max_retries:
                    attempts = attempt + 1
                    raise FridayLLMError(
                        f"Friday native request failed for model {model}: "
                        f"network error after {attempts} attempts: {exc}"
                    ) from exc
            attempt += 1
            if self.retry_backoff_seconds > 0:
                time.sleep(self.retry_backoff_seconds * attempt)

    def cache_key_metadata(self) -> dict[str, Any]:
        return {
            "provider": "friday",
            "native_url": self.native_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": self.response_format,
        }

    def _respect_min_interval(self, model: str) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._rate_limit_lock:
            now = time.monotonic()
            last_request_at = self._last_request_at_by_model.get(model)
            if last_request_at is not None:
                wait_seconds = last_request_at + self.min_interval_seconds - now
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                    now = time.monotonic()
            self._last_request_at_by_model[model] = now

    @staticmethod
    def _load_api_key_provider() -> Callable[[str], str]:
        def provider(model: str) -> str:
            key = os.getenv("COPAL_FRIDAY_API_KEY", os.getenv("COPAL_OPENAI_COMPATIBLE_API_KEY", "")).strip()
            key_file = os.getenv(
                "COPAL_FRIDAY_API_KEY_FILE",
                os.getenv("COPAL_OPENAI_COMPATIBLE_API_KEY_FILE", ""),
            ).strip()
            if key and key_file:
                raise ValueError("Set only one of COPAL_FRIDAY_API_KEY and COPAL_FRIDAY_API_KEY_FILE")
            if key:
                return key
            if key_file:
                payload = read_json(Path(key_file))
                for field_name in ("COPAL_FRIDAY_API_KEY", "OPENAI_COMPATIBLE_API_KEY", "api_key"):
                    value = payload.get(field_name)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                raise ValueError(
                    f"{key_file} must contain COPAL_FRIDAY_API_KEY, OPENAI_COMPATIBLE_API_KEY, or api_key"
                )
            try:
                from llmcore_sdk.models.friday import _get_api_key
            except ImportError as exc:  # pragma: no cover - depends on local internal SDK
                raise ImportError(
                    "No OpenAI-compatible API key is configured. Set COPAL_FRIDAY_API_KEY, "
                    "COPAL_OPENAI_COMPATIBLE_API_KEY, COPAL_FRIDAY_API_KEY_FILE, or "
                    "COPAL_OPENAI_COMPATIBLE_API_KEY_FILE before using this provider."
                ) from exc
            return _get_api_key(model)

        return provider

    @staticmethod
    def _load_post() -> Callable[..., Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - environment dependency check
            raise ImportError(
                "requests is required for Friday native live execution. "
                "Install it in the COPAL environment before using --execution-mode live."
            ) from exc
        return requests.post

    @staticmethod
    def _load_retryable_network_errors() -> tuple[type[BaseException], ...]:
        try:
            from requests.exceptions import ConnectionError, Timeout
        except ImportError:  # pragma: no cover - requests is a live dependency
            return ()
        return (ConnectionError, Timeout)

    @staticmethod
    def _is_retryable_friday_error(exc: FridayLLMError) -> bool:
        if exc.status_code == 429 or (exc.status_code is not None and 500 <= exc.status_code <= 599):
            return True
        if "Access to Bedrock models is not allowed for this account" in str(exc):
            return True
        return "returned empty content" in str(exc)

    @staticmethod
    def _native_body(*, native_response: Any, model: str) -> dict[str, Any]:
        status_code = int(native_response.status_code)
        try:
            body = native_response.json()
        except ValueError as exc:
            raise FridayLLMError(
                f"Friday native request failed for model {model}: HTTP {status_code}; "
                f"response was not valid JSON: {native_response.text}",
                status_code=status_code,
            ) from exc
        if not isinstance(body, dict):
            raise FridayLLMError(
                f"Friday native request failed for model {model}: HTTP {status_code}; "
                f"JSON body was not an object",
                status_code=status_code,
            )
        if status_code < 200 or status_code >= 300:
            raise FridayLLMError(
                f"Friday native request failed for model {model}: HTTP {status_code}; body={body}",
                status_code=status_code,
            )
        return body

    @staticmethod
    def _response_from_native_body(*, body: dict[str, Any], requested_model: str) -> LLMResponse:
        choices = body["choices"]
        if not isinstance(choices, list) or not choices:
            raise FridayLLMError(f"Friday native response for model {requested_model} has no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise FridayLLMError(f"Friday native response for model {requested_model} has an invalid first choice")
        message = first_choice["message"]
        if not isinstance(message, dict):
            raise FridayLLMError(f"Friday native response for model {requested_model} has an invalid message")
        content = message["content"]
        if not isinstance(content, str):
            raise FridayLLMError(f"Friday native response for model {requested_model} returned non-string content")
        if not content.strip():
            raise FridayLLMError(f"Friday native response for model {requested_model} returned empty content")
        usage = body["usage"]
        if not isinstance(usage, dict):
            raise FridayLLMError(f"Friday native response for model {requested_model} has invalid usage metadata")
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        total_tokens = int(usage["total_tokens"])
        return LLMResponse(
            text=content,
            model=str(body["model"]),
            provider="friday",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            raw_response=body,
        )

    def _record_usage(self, response: LLMResponse) -> None:
        _record_response_usage(self, response)

    def usage_summary(self) -> dict[str, Any]:
        with self._usage_lock:
            return {
                "provider": "friday",
                "direction": self.direction,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "last_cost": self.last_cost,
                "total_cost": self.total_cost,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            }


class OpenRouterLLMClient:
    """OpenRouter OpenAI-compatible client used for externally routed models."""

    def __init__(
        self,
        *,
        model_map: dict[str, str],
        max_tokens: int = 8000,
        temperature: float = 1.0,
        timeout: int = 240,
        response_format: dict[str, Any] | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        api_key_provider: Callable[[], str] | None = None,
        post: Callable[..., Any] | None = None,
        native_url: str = OPENROUTER_CHAT_URL,
    ) -> None:
        if not model_map:
            raise ValueError("OpenRouter model_map must not be empty")
        self.model_map = dict(model_map)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.response_format = response_format
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.native_url = native_url
        self._api_key_provider = api_key_provider or _openrouter_api_key_provider_from_env
        self._post = post or FridayLLMClient._load_post()
        self._retryable_network_errors = FridayLLMClient._load_retryable_network_errors()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.last_cost: float | None = None
        self.total_cost: float | None = None
        self.total_prompt_tokens: int | None = None
        self.total_completion_tokens: int | None = None
        self.last_response: Any | None = None
        self._usage_lock = Lock()

    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        if model not in self.model_map:
            raise OpenRouterLLMError(f"OpenRouter route is not configured for model {model}")
        provider_model = self.model_map[model]
        payload: dict[str, Any] = {
            "model": provider_model,
            "messages": [asdict(message) for message in messages],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.response_format is not None:
            payload["response_format"] = self.response_format

        attempt = 0
        while True:
            try:
                response = self._post(
                    url=self.native_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                body = self._native_body(native_response=response, requested_model=model, provider_model=provider_model)
                llm_response = self._response_from_native_body(body=body, requested_model=model)
                _record_response_usage(self, llm_response)
                return llm_response
            except OpenRouterLLMError as exc:
                if not self._is_retryable_openrouter_error(exc) or attempt >= self.max_retries:
                    raise
            except self._retryable_network_errors as exc:
                if attempt >= self.max_retries:
                    attempts = attempt + 1
                    raise OpenRouterLLMError(
                        f"OpenRouter request failed for model {model}: "
                        f"network error after {attempts} attempts: {exc}"
                    ) from exc
            attempt += 1
            if self.retry_backoff_seconds > 0:
                time.sleep(self.retry_backoff_seconds * attempt)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key_provider()}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("COPAL_OPENROUTER_REFERER", "https://github.com/bruceallen/COPAL"),
            "X-Title": os.getenv("COPAL_OPENROUTER_TITLE", "COPAL"),
        }

    def cache_key_metadata(self) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "native_url": self.native_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": self.response_format,
            "model_map": self.model_map,
        }

    @staticmethod
    def _is_retryable_openrouter_error(exc: OpenRouterLLMError) -> bool:
        return exc.status_code == 429 or (exc.status_code is not None and 500 <= exc.status_code <= 599)

    @staticmethod
    def _native_body(*, native_response: Any, requested_model: str, provider_model: str) -> dict[str, Any]:
        status_code = int(native_response.status_code)
        try:
            body = native_response.json()
        except ValueError as exc:
            raise OpenRouterLLMError(
                f"OpenRouter request failed for model {requested_model} via {provider_model}: HTTP {status_code}; "
                f"response was not valid JSON: {native_response.text}",
                status_code=status_code,
            ) from exc
        if not isinstance(body, dict):
            raise OpenRouterLLMError(
                f"OpenRouter request failed for model {requested_model} via {provider_model}: HTTP {status_code}; "
                "JSON body was not an object",
                status_code=status_code,
            )
        if status_code < 200 or status_code >= 300:
            raise OpenRouterLLMError(
                f"OpenRouter request failed for model {requested_model} via {provider_model}: "
                f"HTTP {status_code}; body={body}",
                status_code=status_code,
            )
        return body

    @staticmethod
    def _response_from_native_body(*, body: dict[str, Any], requested_model: str) -> LLMResponse:
        choices = body["choices"]
        if not isinstance(choices, list) or not choices:
            raise OpenRouterLLMError(f"OpenRouter response for model {requested_model} has no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenRouterLLMError(f"OpenRouter response for model {requested_model} has an invalid first choice")
        message = first_choice["message"]
        if not isinstance(message, dict):
            raise OpenRouterLLMError(f"OpenRouter response for model {requested_model} has an invalid message")
        content = message["content"]
        if not isinstance(content, str):
            raise OpenRouterLLMError(f"OpenRouter response for model {requested_model} returned non-string content")
        if not content.strip():
            raise OpenRouterLLMError(f"OpenRouter response for model {requested_model} returned empty content")
        usage = body["usage"]
        if not isinstance(usage, dict):
            raise OpenRouterLLMError(f"OpenRouter response for model {requested_model} has invalid usage metadata")
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        total_tokens = int(usage["total_tokens"])
        return LLMResponse(
            text=content,
            model=str(body["model"]),
            provider="openrouter",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            raw_response=body,
        )

    def usage_summary(self) -> dict[str, Any]:
        with self._usage_lock:
            return {
                "provider": "openrouter",
                "routed_models": dict(self.model_map),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "last_cost": self.last_cost,
                "total_cost": self.total_cost,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            }


class ModelRoutingLLMClient:
    """Route selected model names to provider-specific clients, otherwise use default_client."""

    def __init__(
        self,
        *,
        default_client: LLMClient,
        routes: dict[str, tuple[LLMClient, dict[str, str]]],
    ) -> None:
        self.default_client = default_client
        self.routes = {provider: (client, dict(model_map)) for provider, (client, model_map) in routes.items()}

    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        for client, model_map in self.routes.values():
            if model in model_map:
                return client.complete(model=model, messages=messages)
        return self.default_client.complete(model=model, messages=messages)

    def cache_key_metadata(self) -> dict[str, Any]:
        return {
            "provider": "routed",
            "default": self._client_cache_key_metadata(self.default_client),
            "routes": {
                provider: {
                    "models": dict(model_map),
                    "generation": self._client_cache_key_metadata(client),
                }
                for provider, (client, model_map) in self.routes.items()
            },
        }

    @staticmethod
    def _client_cache_key_metadata(client: LLMClient) -> dict[str, Any]:
        metadata = client.cache_key_metadata
        if not callable(metadata):
            raise TypeError("Routed LLM clients must expose cache_key_metadata")
        value = metadata()
        if not isinstance(value, dict):
            raise TypeError("Routed LLM client cache_key_metadata must return a dict")
        return value

    def usage_summary(self) -> dict[str, Any]:
        default_usage = self._client_usage_summary(self.default_client)
        route_usage = {
            provider: self._client_usage_summary(client)
            for provider, (client, _model_map) in self.routes.items()
        }
        all_usage = [default_usage, *route_usage.values()]
        return {
            "provider": "routed",
            "default_provider": str(default_usage["provider"]),
            "routes": {
                provider: {
                    "models": dict(model_map),
                    "usage": route_usage[provider],
                }
                for provider, (_client, model_map) in self.routes.items()
            },
            "prompt_tokens": sum(int(usage.get("prompt_tokens", 0) or 0) for usage in all_usage),
            "completion_tokens": sum(int(usage.get("completion_tokens", 0) or 0) for usage in all_usage),
            "total_tokens": sum(int(usage.get("total_tokens", 0) or 0) for usage in all_usage),
        }

    @staticmethod
    def _client_usage_summary(client: LLMClient) -> dict[str, Any]:
        summary = client.usage_summary
        if not callable(summary):
            raise TypeError("Routed LLM clients must expose usage_summary")
        value = summary()
        if not isinstance(value, dict):
            raise TypeError("Routed LLM client usage_summary must return a dict")
        return value


class CachedLLMClient:
    def __init__(self, *, base_client: LLMClient, cache_dir: Path) -> None:
        self._base_client = base_client
        self._cache_dir = ensure_directory(cache_dir)
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache_lock = Lock()

    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        payload = {
            "model": model,
            "messages": [asdict(message) for message in messages],
            "generation": self._cache_key_metadata(),
        }
        digest = sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = self._cache_dir / f"{digest}.json"
        with self._cache_lock:
            if cache_path.exists():
                self.cache_hits += 1
                cached = read_json(cache_path)
                return self._response_from_cache(cached)
            self.cache_misses += 1
        response = self._base_client.complete(model=model, messages=messages)
        with self._cache_lock:
            if cache_path.exists():
                cached = read_json(cache_path)
                return self._response_from_cache(cached)
            write_json(cache_path, self._response_to_cache(response))
        return response

    def _cache_key_metadata(self) -> dict[str, Any]:
        metadata = self._base_client.cache_key_metadata
        if not callable(metadata):
            raise TypeError("CachedLLMClient base_client.cache_key_metadata must be callable")
        value = metadata()
        if not isinstance(value, dict):
            raise TypeError("CachedLLMClient base_client.cache_key_metadata must return a dict")
        return value

    @staticmethod
    def _response_to_cache(response: LLMResponse) -> dict[str, Any]:
        return {
            "text": response.text,
            "model": response.model,
            "provider": response.provider,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "last_cost": response.last_cost,
            "total_cost": response.total_cost,
            "total_prompt_tokens": response.total_prompt_tokens,
            "total_completion_tokens": response.total_completion_tokens,
        }

    @staticmethod
    def _response_from_cache(cached: dict[str, Any]) -> LLMResponse:
        return LLMResponse(
            text=str(cached["text"]),
            model=str(cached["model"]),
            provider="cache",
            prompt_tokens=int(cached["prompt_tokens"]),
            completion_tokens=int(cached["completion_tokens"]),
            total_tokens=int(cached["total_tokens"]),
            last_cost=cached["last_cost"],
            total_cost=cached["total_cost"],
            total_prompt_tokens=cached["total_prompt_tokens"],
            total_completion_tokens=cached["total_completion_tokens"],
        )

    def usage_summary(self) -> dict[str, Any]:
        summary = self._base_client.usage_summary
        if not callable(summary):
            raise TypeError("CachedLLMClient base_client.usage_summary must be callable")
        usage = dict(summary())
        with self._cache_lock:
            usage["cache_hits"] = self.cache_hits
            usage["cache_misses"] = self.cache_misses
        usage["cache_dir"] = str(self._cache_dir)
        return usage


def parse_strict_json_payload(text: str) -> Any:
    stripped = text.strip()
    last_error: json.JSONDecodeError | None = None
    for candidate in _json_payload_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise LLMJsonError("LLM response was not strict JSON or a supported JSON wrapper", response_text=text) from last_error
    raise LLMJsonError("LLM response was not strict JSON or a supported JSON wrapper", response_text=text)


def _json_payload_candidates(stripped: str) -> list[str]:
    candidates = [stripped]
    fenced = _strip_markdown_json_fence(stripped)
    if fenced is not None:
        candidates.append(fenced)
    without_think = _strip_leading_think_block(stripped)
    if without_think is not None:
        candidates.append(without_think)
        fenced_without_think = _strip_markdown_json_fence(without_think)
        if fenced_without_think is not None:
            candidates.append(fenced_without_think)
    suffix_after_think = _strip_suffix_after_think_close(stripped)
    if suffix_after_think is not None:
        candidates.append(suffix_after_think)
        fenced_suffix_after_think = _strip_markdown_json_fence(suffix_after_think)
        if fenced_suffix_after_think is not None:
            candidates.append(fenced_suffix_after_think)
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _strip_markdown_json_fence(text: str) -> str | None:
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return None
    opening = lines[0].strip().lower()
    if opening not in ("```json", "```"):
        return None
    if lines[-1].strip() != "```":
        return None
    return "\n".join(lines[1:-1]).strip()


def _strip_leading_think_block(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return None
    end_index = stripped.find("</think>")
    if end_index < 0:
        return None
    return stripped[end_index + len("</think>") :].strip()


def _strip_suffix_after_think_close(text: str) -> str | None:
    end_index = text.rfind("</think>")
    if end_index < 0:
        return None
    suffix = text[end_index + len("</think>") :].strip()
    if not suffix:
        return None
    return suffix

def complete_json(*, client: LLMClient, model: str, messages: list[LLMMessage]) -> Any:
    response = client.complete(model=model, messages=messages)
    return parse_strict_json_payload(response.text)


def _friday_response_format_from_env() -> dict[str, Any] | None:
    value = os.getenv("COPAL_FRIDAY_RESPONSE_FORMAT", "").strip()
    if not value:
        return None
    if value == "json_object":
        return {"type": "json_object"}
    raise ValueError(
        "Unsupported COPAL_FRIDAY_RESPONSE_FORMAT: "
        f"{value}. Supported value: json_object"
    )


def _openrouter_response_format_from_env() -> dict[str, Any] | None:
    value = os.getenv("COPAL_OPENROUTER_RESPONSE_FORMAT", "").strip()
    if value:
        if value == "json_object":
            return {"type": "json_object"}
        raise ValueError(
            "Unsupported COPAL_OPENROUTER_RESPONSE_FORMAT: "
            f"{value}. Supported value: json_object"
        )
    return _friday_response_format_from_env()


def _openrouter_api_key_provider_from_env() -> str:
    key = os.getenv("COPAL_OPENROUTER_API_KEY", "").strip()
    key_file = os.getenv("COPAL_OPENROUTER_API_KEY_FILE", "").strip()
    if key and key_file:
        raise ValueError("Set only one of COPAL_OPENROUTER_API_KEY and COPAL_OPENROUTER_API_KEY_FILE")
    if key:
        return key
    if key_file:
        payload = read_json(Path(key_file))
        value = payload["OPENROUTER_API_KEY"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"OPENROUTER_API_KEY is empty or invalid in {key_file}")
        return value.strip()
    raise ValueError(
        "OpenRouter route selected but no API key is configured. "
        "Set COPAL_OPENROUTER_API_KEY or COPAL_OPENROUTER_API_KEY_FILE."
    )


def _openrouter_key_is_configured() -> bool:
    return bool(os.getenv("COPAL_OPENROUTER_API_KEY", "").strip() or os.getenv("COPAL_OPENROUTER_API_KEY_FILE", "").strip())


def _openrouter_api_key_source_metadata() -> dict[str, str]:
    key = os.getenv("COPAL_OPENROUTER_API_KEY", "").strip()
    key_file = os.getenv("COPAL_OPENROUTER_API_KEY_FILE", "").strip()
    if key and key_file:
        raise ValueError("Set only one of COPAL_OPENROUTER_API_KEY and COPAL_OPENROUTER_API_KEY_FILE")
    if key:
        return {"source": "env:COPAL_OPENROUTER_API_KEY", "fingerprint": sha256(key.encode("utf-8")).hexdigest()[:12]}
    if key_file:
        secret = _openrouter_api_key_provider_from_env()
        return {
            "source": f"file:{key_file}:OPENROUTER_API_KEY",
            "fingerprint": sha256(secret.encode("utf-8")).hexdigest()[:12],
        }
    return {"source": "unconfigured", "fingerprint": ""}


def _validate_model_map(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object mapping COPAL model names to provider model ids")
    model_map: dict[str, str] = {}
    for source_model, provider_model in value.items():
        if not isinstance(source_model, str) or not source_model.strip():
            raise ValueError(f"{context} contains an invalid COPAL model name")
        if not isinstance(provider_model, str) or not provider_model.strip():
            raise ValueError(f"{context} contains an invalid provider model id for {source_model}")
        model_map[source_model.strip()] = provider_model.strip()
    return model_map


def openrouter_model_map_from_env() -> dict[str, str]:
    value = os.getenv("COPAL_OPENROUTER_MODEL_MAP", "").strip()
    if value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("COPAL_OPENROUTER_MODEL_MAP must be valid JSON") from exc
        return _validate_model_map(parsed, context="COPAL_OPENROUTER_MODEL_MAP")
    if _openrouter_key_is_configured():
        return dict(DEFAULT_OPENROUTER_MODEL_MAP)
    return {}


def live_provider_mode_from_env() -> str:
    value = os.getenv("COPAL_LIVE_PROVIDER", "auto").strip().lower()
    if value not in LIVE_PROVIDER_MODES:
        allowed = ", ".join(LIVE_PROVIDER_MODES)
        raise ValueError(f"Unsupported COPAL_LIVE_PROVIDER: {value}. Expected one of: {allowed}")
    return value


def friday_client_config_from_env(*, response_format_override: object = _USE_ENV_RESPONSE_FORMAT) -> dict[str, Any]:
    response_format = (
        _friday_response_format_from_env()
        if response_format_override is _USE_ENV_RESPONSE_FORMAT
        else response_format_override
    )
    return {
        "direction": os.getenv("COPAL_FRIDAY_DIRECTION", "COPAL"),
        "max_tokens": int(os.getenv("COPAL_FRIDAY_MAX_TOKENS", "8000")),
        "temperature": float(os.getenv("COPAL_FRIDAY_TEMPERATURE", "1")),
        "timeout": int(os.getenv("COPAL_FRIDAY_TIMEOUT", "240")),
        "response_format": response_format,
        "max_retries": int(os.getenv("COPAL_FRIDAY_MAX_RETRIES", "2")),
        "retry_backoff_seconds": float(os.getenv("COPAL_FRIDAY_RETRY_BACKOFF_SECONDS", "1")),
        "min_interval_seconds": float(os.getenv("COPAL_FRIDAY_MIN_INTERVAL_SECONDS", "0")),
    }


def openrouter_client_config_from_env(
    *, model_map: dict[str, str], response_format_override: object = _USE_ENV_RESPONSE_FORMAT
) -> dict[str, Any]:
    response_format = (
        _openrouter_response_format_from_env()
        if response_format_override is _USE_ENV_RESPONSE_FORMAT
        else response_format_override
    )
    return {
        "model_map": model_map,
        "max_tokens": int(os.getenv("COPAL_OPENROUTER_MAX_TOKENS", os.getenv("COPAL_FRIDAY_MAX_TOKENS", "8000"))),
        "temperature": float(os.getenv("COPAL_OPENROUTER_TEMPERATURE", os.getenv("COPAL_FRIDAY_TEMPERATURE", "1"))),
        "timeout": int(os.getenv("COPAL_OPENROUTER_TIMEOUT", os.getenv("COPAL_FRIDAY_TIMEOUT", "240"))),
        "response_format": response_format,
        "max_retries": int(os.getenv("COPAL_OPENROUTER_MAX_RETRIES", os.getenv("COPAL_FRIDAY_MAX_RETRIES", "2"))),
        "retry_backoff_seconds": float(
            os.getenv("COPAL_OPENROUTER_RETRY_BACKOFF_SECONDS", os.getenv("COPAL_FRIDAY_RETRY_BACKOFF_SECONDS", "1"))
        ),
    }


def live_client_runtime_metadata() -> dict[str, Any]:
    provider_mode = live_provider_mode_from_env()
    openrouter_model_map = openrouter_model_map_from_env()
    if provider_mode == "openrouter":
        if not openrouter_model_map:
            raise ValueError(
                "COPAL_LIVE_PROVIDER=openrouter requires COPAL_OPENROUTER_API_KEY or "
                "COPAL_OPENROUTER_API_KEY_FILE, and optionally COPAL_OPENROUTER_MODEL_MAP."
            )
        return {
            "provider": "openrouter",
            "model_map": openrouter_model_map,
            "api_key": _openrouter_api_key_source_metadata(),
            "config": {
                key: value
                for key, value in openrouter_client_config_from_env(model_map=openrouter_model_map).items()
                if key != "model_map"
            },
        }
    if provider_mode == "routed" or (provider_mode == "auto" and openrouter_model_map):
        return {
            "provider": "routed",
            "default_provider": "friday",
            "friday": friday_client_config_from_env(),
            "routes": {
                "openrouter": {
                    "model_map": openrouter_model_map,
                    "api_key": _openrouter_api_key_source_metadata(),
                    "config": {
                        key: value
                        for key, value in openrouter_client_config_from_env(model_map=openrouter_model_map).items()
                        if key != "model_map"
                    },
                }
            },
        }
    if provider_mode == "routed" and not openrouter_model_map:
        raise ValueError(
            "COPAL_LIVE_PROVIDER=routed requires an OpenRouter route. "
            "Set COPAL_OPENROUTER_API_KEY or COPAL_OPENROUTER_API_KEY_FILE."
        )
    return {
        "provider": "friday",
        **friday_client_config_from_env(),
    }


def build_live_client(*, cache_dir: Path, response_format_override: object = _USE_ENV_RESPONSE_FORMAT) -> LLMClient:
    provider_mode = live_provider_mode_from_env()
    openrouter_model_map = openrouter_model_map_from_env()
    if provider_mode == "openrouter":
        if not openrouter_model_map:
            raise ValueError(
                "COPAL_LIVE_PROVIDER=openrouter requires COPAL_OPENROUTER_API_KEY or "
                "COPAL_OPENROUTER_API_KEY_FILE, and optionally COPAL_OPENROUTER_MODEL_MAP."
            )
        base_client: LLMClient = OpenRouterLLMClient(
            **openrouter_client_config_from_env(
                model_map=openrouter_model_map,
                response_format_override=response_format_override,
            )
        )
    elif provider_mode == "routed" or (provider_mode == "auto" and openrouter_model_map):
        if not openrouter_model_map:
            raise ValueError(
                "COPAL_LIVE_PROVIDER=routed requires an OpenRouter route. "
                "Set COPAL_OPENROUTER_API_KEY or COPAL_OPENROUTER_API_KEY_FILE."
            )
        friday_client = FridayLLMClient(**friday_client_config_from_env(response_format_override=response_format_override))
        base_client: LLMClient = ModelRoutingLLMClient(
            default_client=friday_client,
            routes={
                "openrouter": (
                    OpenRouterLLMClient(
                        **openrouter_client_config_from_env(
                            model_map=openrouter_model_map,
                            response_format_override=response_format_override,
                        )
                    ),
                    openrouter_model_map,
                )
            },
        )
    else:
        base_client = FridayLLMClient(**friday_client_config_from_env(response_format_override=response_format_override))
    return CachedLLMClient(
        base_client=base_client,
        cache_dir=cache_dir,
    )
