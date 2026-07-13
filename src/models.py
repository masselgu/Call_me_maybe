from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterDefinition(BaseModel):
    """Describe one parameter accepted by a callable function."""
    model_config = ConfigDict(extra="forbid")
    type: JsonType


class ReturnDefinition(BaseModel):
    """Describe the return value of a function."""
    model_config = ConfigDict(extra="forbid")
    type: JsonType


class FunctionDefinition(BaseModel):
    """Describe an available function and its JSON schema."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("function name cannot be empty")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
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
        for name in value:
            if not name.strip():
                raise ValueError(
                    "parameter names cannot be empty"
                )
        return value


class PromptItem(BaseModel):
    """Input prompt read from the tests file."""
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt cannot be empty")
        return value


class FunctionCallResult(BaseModel):
    """Output record required by the subject."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
