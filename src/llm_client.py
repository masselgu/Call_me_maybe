"""Small wrapper around the provided LLM SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Vocabulary(BaseModel):
    """Token mapping loaded from the model vocabulary JSON file."""

    token_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str) -> "Vocabulary":
        """Load a Hugging Face ``vocab.json`` file."""
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("vocabulary JSON must contain an object")
        token_to_id = {
            str(token): int(token_id)
            for token, token_id in raw.items()
        }
        id_to_token = {
            token_id: token for token, token_id in token_to_id.items()
        }
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def token_text(self, token_id: int) -> str:
        """Return the text associated with one token id."""
        return self.id_to_token.get(token_id, "")

    def has_id(self, token_id: int) -> bool:
        """Check whether a token id exists in the vocabulary."""
        return token_id in self.id_to_token


class LlmClient(BaseModel):
    """Load the provided SDK and expose a narrow typed interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"
    _model: Any = PrivateAttr(default=None)
    vocabulary: Vocabulary = Field(default_factory=Vocabulary)

    def model_post_init(self, __context: Any) -> None:
        """Load the model after Pydantic validates the configuration."""
        try:
            from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

            self._model = Small_LLM_Model(model_name=self.model_name)
            vocab_path: str = self._model.get_path_to_vocab_file()
            self.vocabulary = Vocabulary.from_path(vocab_path)
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

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text."""
        decoded: Any = self._model.decode(token_ids)
        if not isinstance(decoded, str):
            raise ValueError("LLM SDK returned unsupported decoded text")
        return decoded

    def logits(self, input_ids: list[int]) -> list[float]:
        """Return raw next-token logits for the current context."""
        try:
            raw_logits: Any = self._model.get_logits_from_input_ids(input_ids)
        except Exception as exc:
            raise ValueError(f"LLM inference failed: {exc}") from exc
        if not isinstance(raw_logits, list):
            raise ValueError("LLM SDK returned unsupported logits")
        return [float(item) for item in raw_logits]
