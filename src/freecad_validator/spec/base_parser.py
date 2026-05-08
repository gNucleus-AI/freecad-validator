"""Base class + shared schema for spec parsers.

A spec parser turns a loosely-structured spec dict
(``{"name", "description", "key_parameters"}``) into a `StructuredSpec`
with typed scalar / vector / count param dicts. Concrete
implementations pick the extraction strategy (regex rules, LLM
fallback, etc.) and live in sibling modules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class StructuredSpec(BaseModel):
    """Parsed spec. All values in normalized units: mm / rad / int."""

    name: str
    description: str
    scalars: dict[str, float] = Field(default_factory=dict)
    vectors: dict[str, tuple[float, ...]] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    # Populated by the optional LLM fallback. The regex-only
    # path leaves this empty; the classifier must tolerate that.
    expected_features: list[str] = Field(default_factory=list)


class SpecBaseParser(ABC):
    """Turn a spec dict into a `StructuredSpec`."""

    #: Short stable identifier used in logs and error messages.
    name: str = ""

    @abstractmethod
    def parse_spec(self, spec: dict[str, str]) -> StructuredSpec:
        """Parse ``spec`` (envelope: name, description, key_parameters) into
        a StructuredSpec. Missing envelope keys are treated as empty
        strings; subclasses must not raise on partial specs."""
