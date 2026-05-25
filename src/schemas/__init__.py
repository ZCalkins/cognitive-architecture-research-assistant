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
from src.schemas.lifecycle import (
    LifecycleEvent,
    LifecycleManager,
    LifecycleTrigger,
    MergeTrigger,
    PruneTrigger,
    SpawnTrigger,
    SplitTrigger,
)
from src.schemas.types import SpawnRecord, StructureMapping, TypedSlot

__all__ = [
    "CompositeBacking",
    "LLMBacking",
    "LifecycleEvent",
    "LifecycleManager",
    "LifecycleTrigger",
    "MergeTrigger",
    "NeuralBacking",
    "PruneTrigger",
    "Schema",
    "SchemaBacking",
    "SchemaMeta",
    "SpawnRecord",
    "SpawnTrigger",
    "SplitTrigger",
    "StructureMapping",
    "SymbolicBacking",
    "TypedSlot",
]
