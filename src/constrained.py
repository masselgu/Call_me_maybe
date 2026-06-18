"""Token-level constrained decoding over finite valid choices."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.llm_client import LlmClient


class EncodedChoice(BaseModel):
    """A candidate completion and its token sequence."""

    model_config = ConfigDict(frozen=True)

    text: str
    token_ids: tuple[int, ...]


class ConstrainedChoiceDecoder(BaseModel):
    """Decode one value while only allowing valid candidate continuations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LlmClient

    def choose(self, context: str, choices: list[str]) -> str:
        """Choose the candidate with the highest total log-probability.

        Each candidate is scored token-by-token using the model logits.
        Invalid tokens are ignored because scoring is limited to complete
        candidate sequences only.
        """
        if not choices:
            raise ValueError("cannot decode from an empty choice list")
        if len(choices) == 1:
            return choices[0]

        encoded_choices: list[EncodedChoice] = self._encode_choices(choices)
        context_ids: list[int] = self.llm.encode(context)

        best_choice: str = encoded_choices[0].text
        best_score: float = float("-inf")
        for choice in encoded_choices:
            score: float = self._score_sequence(context_ids, choice.token_ids)
            if score > best_score:
                best_score = score
                best_choice = choice.text
        return best_choice

    def _encode_choices(self, choices: list[str]) -> list[EncodedChoice]:
        """Encode choices and make them prefix-safe."""
        seen: set[str] = set()
        encoded: list[EncodedChoice] = []
        for choice in choices:
            if choice in seen:
                continue
            seen.add(choice)
            token_ids: list[int] = self.llm.encode(f"{choice}\n")
            encoded.append(
                EncodedChoice(text=choice, token_ids=tuple(token_ids)),
            )
        return encoded

    def _score_sequence(
        self,
        context_ids: list[int],
        choice_ids: tuple[int, ...],
    ) -> float:
        """Score one complete candidate using summed log-probabilities."""
        ids: list[int] = list(context_ids)
        total: float = 0.0
        for token_id in choice_ids:
            logits: np.ndarray = np.array(
                self.llm.logits(ids),
                dtype=np.float64,
            )
            if token_id >= len(logits):
                return float("-inf")
            shifted: np.ndarray = logits - np.max(logits)
            log_probs: np.ndarray = shifted - np.log(np.sum(np.exp(shifted)))
            total += float(log_probs[token_id])
            ids.append(token_id)
        return total
