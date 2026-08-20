"""FPT AI Marketplace VLM client for the grounded VQA handler."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Protocol

import httpx

from BackEnd.app.vqa.handler import VQAModelAnswer


class HTTPClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...


class FPTVLMClient:
    """Call an OpenAI-compatible vision model on FPT AI Marketplace."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://mkp-api.fptcloud.com",
        timeout_seconds: float = 60.0,
        max_images: int = 5,
        max_image_bytes: int = 5 * 1024 * 1024,
        http_client: HTTPClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_images <= 0 or max_image_bytes <= 0:
            raise ValueError("image limits must be positive")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_images = max_images
        self.max_image_bytes = max_image_bytes
        self.http_client = http_client

    @classmethod
    def from_env(cls, **overrides: Any) -> "FPTVLMClient":
        """Build the client without exposing credentials in source code."""

        api_key = os.getenv("VQA_API_KEY", "")
        model = os.getenv("VQA_MODEL", "")
        base_url = os.getenv("VQA_BASE_URL", "https://mkp-api.fptcloud.com")
        timeout = float(os.getenv("VQA_TIMEOUT_SECONDS", "60"))
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout,
            **overrides,
        )

    def answer(
        self,
        *,
        question: str,
        prompt: str,
        image_paths: list[Path] | tuple[Path, ...],
    ) -> VQAModelAnswer:
        selected_paths = tuple(image_paths[: self.max_images])
        if not selected_paths:
            raise ValueError("At least one evidence image is required")

        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_data_url(path)},
            }
            for path in selected_paths
        ]
        content.append({"type": "text", "text": question})
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{prompt}\nReturn only valid JSON with this schema: "
                        '{"answer":"...","confidence":0.0}. '
                        "confidence must be between 0 and 1."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        response = self._post(payload)
        try:
            response.raise_for_status()
            body = response.json()
            model_content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError("FPT VLM returned an invalid response") from error
        return self._parse_model_answer(model_content)

    def _post(self, payload: dict[str, Any]) -> Any:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        if self.http_client is not None:
            return self.http_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(url, headers=headers, json=payload)

    def _image_data_url(self, image_path: Path) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"VQA evidence image does not exist: {path}")
        size = path.stat().st_size
        if size > self.max_image_bytes:
            raise ValueError(
                f"VQA evidence image exceeds {self.max_image_bytes} bytes: {path}"
            )
        mime_type, _encoding = mimetypes.guess_type(path.name)
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"Unsupported VQA evidence image type: {path.suffix}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _parse_model_answer(content: Any) -> VQAModelAnswer:
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("FPT VLM returned empty content")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(cleaned)
            answer = str(parsed["answer"]).strip()
            confidence = float(parsed["confidence"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("FPT VLM did not return the required JSON") from error
        if not answer:
            raise RuntimeError("FPT VLM returned an empty answer")
        return VQAModelAnswer(
            answer=answer,
            confidence=min(max(confidence, 0.0), 1.0),
        )


__all__ = ["FPTVLMClient"]
