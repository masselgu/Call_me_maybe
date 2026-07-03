from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterDefinition(BaseModel):
    """Describe one parameter accepted by a callable function."""
    type: JsonType


class ReturnDefinition(BaseModel):
    """Describe the return value of a function."""
    type: JsonType


class FunctionDefinition(BaseModel):
    """Describe an available function and its JSON schema."""
    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject empty function names."""
        stripped: str = value.strip()
        if not stripped:
            raise ValueError("function name cannot be empty")
        return stripped


class PromptItem(BaseModel):
    """Input prompt read from the tests file."""
    prompt: str


class FunctionCallResult(BaseModel):
    """Output record required by the subject."""
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
