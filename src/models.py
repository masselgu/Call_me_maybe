from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


JsonType = Literal["string", "number", "integer", "boolean", "array", "object"]


class ParameterDefinition(BaseModel):
    """One parameter accepted by a callable function.

    Attributes:
        type: JSON schema type of the parameter.
    """
    model_config = ConfigDict(extra="forbid")
    type: JsonType


class ReturnDefinition(BaseModel):
    """Return value of a function.

    Attributes:
        type: JSON schema type of the return value.
    """
    model_config = ConfigDict(extra="forbid")
    type: JsonType


class FunctionDefinition(BaseModel):
    """An available function and its JSON schema.

    Attributes:
        name: Function name.
        description: Human-readable description.
        parameters: Map of parameter names to their definitions.
        returns: Return value definition.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject empty function names.

        Args:
            value: The function name to validate.

        Returns:
            Stripped function name.

        Raises:
            ValueError: If name is empty.
        """
        value = value.strip()
        if not value:
            raise ValueError("function name cannot be empty")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        """Reject empty descriptions.

        Args:
            value: The description to validate.

        Returns:
            Stripped description.

        Raises:
            ValueError: If description is empty.
        """
        value = value.strip()
        if not value:
            raise ValueError("function description cannot be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, ParameterDefinition],
    ) -> dict[str, ParameterDefinition]:
        """Validate parameter names are non-empty identifiers.

        Args:
            value: Map of parameter names to definitions.

        Returns:
            Validated parameters dict.

        Raises:
            ValueError: If any name is empty or not a valid identifier.
        """
        for name in value:
            if not name.strip():
                raise ValueError(
                    "parameter names cannot be empty"
                )
            if not name.isidentifier():
                raise ValueError(
                    "parameter names must be valid identifiers"
                )
        return value


class PromptItem(BaseModel):
    """Input prompt read from the tests file.

    Attributes:
        prompt: The natural language request string.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject empty prompts.

        Args:
            value: The prompt to validate.

        Returns:
            Stripped prompt string.

        Raises:
            ValueError: If prompt is empty.
        """
        value = value.strip()
        if not value:
            raise ValueError("prompt cannot be empty")
        return value


class FunctionCallResult(BaseModel):
    """Output record for one processed prompt.

    Attributes:
        prompt: Original natural language request.
        name: Selected function name.
        parameters: Extracted parameter values.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
