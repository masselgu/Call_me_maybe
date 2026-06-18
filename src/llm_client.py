"""Small wrapper around the provided LLM SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr


class LlmClient(BaseModel):
    """Load the provided SDK and expose a narrow typed interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"
    _model: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Load the model after Pydantic validates the configuration."""
        try:
            from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

            self._model = Small_LLM_Model(model_name=self.model_name)
        except Exception as exc:
            message: str = (
                "failed to load the LLM model. Run `uv sync` first and make "
                "sure the model can be downloaded or is available locally."
            )
            raise ValueError(message) from exc

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        encoded: Any = self._model.encode(text)
        if hasattr(encoded, "tolist"):
            raw: Any = encoded.tolist()
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                return [int(item) for item in raw[0]]
            if isinstance(raw, list):
                return [int(item) for item in raw]
        if isinstance(encoded, list):
            return [int(item) for item in encoded]
        raise ValueError("LLM SDK returned unsupported token ids")

    def logits(self, input_ids: list[int]) -> list[float]:
        """Return raw next-token logits for the current context."""
        try:
            raw_logits: Any = self._model.get_logits_from_input_ids(input_ids)
        except Exception as exc:
            raise ValueError(f"LLM inference failed: {exc}") from exc
        if not isinstance(raw_logits, list):
            raise ValueError("LLM SDK returned unsupported logits")
        return [float(item) for item in raw_logits]
