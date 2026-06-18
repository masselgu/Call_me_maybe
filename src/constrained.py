"""Token-level constrained decoding over finite valid choices."""

from __future__ import annotations

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
        """Choose one string by constraining every generated token.

        The model still decides each token from its logits. Impossible
        continuations are removed. The choices are suffixed with a newline
        sentinel internally so every completion has a clear end.
        """
        if not choices:
            raise ValueError("cannot decode from an empty choice list")

        encoded_choices: list[EncodedChoice] = self._encode_choices(choices)
        context_ids: list[int] = self.llm.encode(context)
        generated: list[int] = []

        while True:
            exact: EncodedChoice | None
            exact = self._find_exact(encoded_choices, generated)
            if exact is not None:
                return exact.text

            valid_tokens: set[int]
            valid_tokens = self._valid_next_tokens(encoded_choices, generated)
            if not valid_tokens:
                message: str = "constrained decoding reached a dead end"
                raise ValueError(message)

            logits: list[float] = self.llm.logits(context_ids + generated)
            generated.append(self._best_valid_token(logits, valid_tokens))

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

    @staticmethod
    def _find_exact(
        choices: list[EncodedChoice],
        generated: list[int],
    ) -> EncodedChoice | None:
        """Return the completed choice, if generation is complete."""
        generated_tuple: tuple[int, ...] = tuple(generated)
        for choice in choices:
            if choice.token_ids == generated_tuple:
                return choice
        return None

    @staticmethod
    def _valid_next_tokens(
        choices: list[EncodedChoice],
        generated: list[int],
    ) -> set[int]:
        """Find every token that keeps at least one choice valid."""
        prefix: tuple[int, ...] = tuple(generated)
        valid: set[int] = set()
        for choice in choices:
            if choice.token_ids[: len(prefix)] == prefix:
                if len(choice.token_ids) > len(prefix):
                    valid.add(choice.token_ids[len(prefix)])
        return valid

    @staticmethod
    def _best_valid_token(logits: list[float], valid_tokens: set[int]) -> int:
        """Pick the highest-logit token among valid next tokens."""
        best_token: int | None = None
        best_score: float = float("-inf")
        for token_id in valid_tokens:
            if token_id >= len(logits):
                continue
            score: float = logits[token_id]
            if best_token is None or score > best_score:
                best_token = token_id
                best_score = score
        if best_token is None:
            raise ValueError("no valid token was present in model logits")
        return best_token
