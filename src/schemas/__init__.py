"""Polymorphic schema framework — the universal unit of the architecture.

See ARCHITECTURE.md "Polymorphic schema design" and CLAUDE.md principle 5
("the self-schema is a schema").
"""

from src.schemas.backings import (
    CompositeBacking,
    LLMBacking,
    NeuralBacking,
    SchemaBacking,
    SymbolicBacking,
)
from src.schemas.base import Schema, SchemaMeta
from src.schemas.types import SpawnRecord, StructureMapping, TypedSlot

__all__ = [
    "CompositeBacking",
    "LLMBacking",
    "NeuralBacking",
    "Schema",
    "SchemaBacking",
    "SchemaMeta",
    "SpawnRecord",
    "StructureMapping",
    "SymbolicBacking",
    "TypedSlot",
]
