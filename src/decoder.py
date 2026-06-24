"""Constrained choice decoder built on the provided LLM SDK."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Vocabulary(BaseModel):
    """Token ids loaded from the model vocabulary JSON file."""

    ids: set[int] = Field(default_factory=set)

    @classmethod
    def from_path(cls, path: str) -> "Vocabulary":
        """Load token ids from a Hugging Face vocab JSON file."""
        with open(path, "r", encoding="utf-8") as file:
            raw: Any = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError("vocabulary JSON must contain an object")
        return cls(ids={int(token_id) for token_id in raw.values()})

    def has_id(self, token_id: int) -> bool:
        """Check whether a token id exists in the vocabulary."""
        return token_id in self.ids


class ChoiceDecoder(BaseModel):
    """Generate one valid choice token-by-token with masked logits."""

    model: Any
    vocabulary: Vocabulary

    def choose(self, context: str, choices: list[str]) -> str:
        """Ask the LLM to choose one item while masking invalid tokens."""
        if not choices:
            raise ValueError("empty choice list")
        if len(choices) == 1:
            return choices[0]

        encoded = [
            (choice, self._encode(f"{choice}\n"))
            for choice in dict.fromkeys(choices)
        ]
        ctx = self._encode(context)
        out: list[int] = []

        while True:
            for choice, ids in encoded:
                if ids == out:
                    return choice
            valid = self._next_tokens(encoded, out)
            out.append(self._best_token(ctx + out, valid))

    def _next_tokens(
        self,
        encoded: list[tuple[str, list[int]]],
        prefix: list[int],
    ) -> set[int]:
        valid: set[int] = set()
        size = len(prefix)
        for _, ids in encoded:
            if ids[:size] == prefix and len(ids) > size:
                token_id = ids[size]
                if self.vocabulary.has_id(token_id):
                    valid.add(token_id)
        if not valid:
            raise ValueError("constrained decoding reached a dead end")
        return valid

    def _best_token(self, ctx: list[int], valid: set[int]) -> int:
        logits = np.array(
            self.model.get_logits_from_input_ids(ctx),
            dtype=np.float64,
        )
        masked = np.full_like(logits, float("-inf"))
        for token_id in valid:
            if token_id < len(logits):
                masked[token_id] = logits[token_id]
        best = int(np.argmax(masked))
        if best not in valid:
            raise ValueError("no valid token was present in model logits")
        return best

    def _encode(self, text: str) -> list[int]:
        encoded: Any = self.model.encode(text)
        raw: Any = encoded.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return [int(item) for item in raw[0]]
        if isinstance(raw, list):
            return [int(item) for item in raw]
        raise ValueError("LLM SDK returned unsupported token ids")
