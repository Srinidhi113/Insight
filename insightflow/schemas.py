from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Options(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['ai', 'demo'] = 'ai'
    num_topics: int = Field(3, ge=1, le=12)
    chunk_tokens: int = Field(700, ge=64, le=900)
    semantic_refinement: bool = True
    dedup_threshold: float = Field(.92, ge=.75, le=1)
    second_pass: bool = True
    summary_tokens: int = Field(200, ge=40, le=256)

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str | None = Field(None, max_length=250_000)
    discussion: dict[str, Any] | list[Any] | None = None
    options: Options = Field(default_factory=Options)

    @model_validator(mode='after')
    def one_input(self):
        if (self.text is None) == (self.discussion is None):
            raise ValueError('Provide exactly one of text or discussion.')
        return self

class SummarizeRequest(BaseModel):
    sentences: list[str] = Field(min_length=1, max_length=1500)
    options: Options = Field(default_factory=Options)
