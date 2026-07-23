import json
from typing import Any
import numpy as np
from pydantic import BaseModel


class Vocabulary(BaseModel):
    """Token vocabulary loaded from the model's vocab file.

    Attributes:
        ids: Set of all valid token IDs.
    """
    ids: set[int]

    @classmethod
    def from_path(cls, path: str) -> "Vocabulary":
        """Load vocabulary from a JSON file.

        Args:
            path: Path to the vocab.json file.

        Returns:
            Vocabulary instance with all token IDs.
        """
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)

        return cls(
            ids={int(token_id) for token_id in raw.values()}
        )

    def has_id(self, token_id: int) -> bool:
        """Check if a token ID exists in the vocabulary.

        Args:
            token_id: The token ID to check.

        Returns:
            True if the token ID is valid.
        """
        return token_id in self.ids


class ChoiceDecoder(BaseModel):
    """Constrained decoder that selects from a finite set of choices.

    Attributes:
        model: The LLM model instance.
        vocabulary: The token vocabulary.
    """
    model: Any
    vocabulary: Vocabulary

    def choose_function(self, context: str, choices: list[str]) -> str:
        """Select the best matching choice using constrained decoding.

        Masks all tokens that are not valid continuations of any
        choice to -inf, guaranteeing the output is always one of
        the provided choices.

        Args:
            context: The prompt context string.
            choices: List of valid string choices to select from.

        Returns:
            The selected choice string.
        """
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
                    is_prefix = any(
                        other_ids[:len(ids)] == ids
                        and len(other_ids) > len(ids)
                        for other_choice, other_ids in encoded_choices
                        if other_choice != choice
                    )
                    if not is_prefix:
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

            masked = np.full_like(logits, float("-inf"))

            for token_id in valid:
                if self.vocabulary.has_id(token_id):
                    masked[token_id] = logits[token_id]
            generated.append(int(np.argmax(masked)))

        raise ValueError("no valid choice found")

    def _encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text: The text to encode.

        Returns:
            List of integer token IDs.

        Raises:
            ValueError: If tokenization produces unexpected format.
        """
        encoded = self.model.encode(text)
        raw = encoded.tolist()
        if isinstance(raw, list):
            if raw and isinstance(raw[0], list):
                return [int(x) for x in raw[0]]

        raise ValueError("invalid tokenization")
