"""Simple constrained decoder."""

import json
from typing import Any
import numpy as np
from pydantic import BaseModel, Field


class Vocabulary(BaseModel):
    ids: set[int] = Field(default_factory=set)

    @classmethod
    def from_path(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)

        return cls(
            ids={int(token_id) for token_id in raw.values()}
        )

    def has_id(self, token_id: int) -> bool:
        return token_id in self.ids


class ChoiceDecoder(BaseModel):
    model: Any
    vocabulary: Vocabulary

    def choose(
        self,
        context: str,
        choices: list[str],
    ) -> str:

        if len(choices) == 1:
            return choices[0]

        encoded_choices = [
            (choice, self._encode(choice))
            for choice in choices
        ]

        context_ids = self._encode(context)
        generated: list[int] = []

        while True:

            for choice, ids in encoded_choices:
                if generated == ids:
                    return choice

            valid: set[int] = set()

            for _, ids in encoded_choices:
                if (
                    ids[:len(generated)] == generated
                    and len(ids) > len(generated)
                ):
                    valid.add(ids[len(generated)])

            logits = np.array(
                self.model.get_logits_from_input_ids(
                    context_ids + generated
                ),
                dtype=np.float64,
            )

            masked = np.full_like(
                logits,
                float("-inf"),
            )

            for token_id in valid:
                if token_id < len(logits):
                    masked[token_id] = logits[token_id]

            generated.append(int(np.argmax(masked)))

    def _encode(self, text: str) -> list[int]:

        encoded = self.model.encode(text)
        raw = encoded.tolist()

        if isinstance(raw, list):

            if raw and isinstance(raw[0], list):
                return [int(x) for x in raw[0]]

            return [int(x) for x in raw]

        raise ValueError("invalid tokenization")
