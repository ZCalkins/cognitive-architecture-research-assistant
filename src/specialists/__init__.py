"""Seed specialists — real LLMBacking-backed assessment schemas (Session 4).

See ARCHITECTURE.md "Phase 1 deliverable 3". Each specialist is an object-level
oracle; the orchestrator decides, the specialists assess (the (C) firewall).
"""

from src.specialists.seed_specialists import (
    SPECIALIST_NAMES,
    build_specialist,
    build_v0_specialist_registry,
)

__all__ = [
    "SPECIALIST_NAMES",
    "build_specialist",
    "build_v0_specialist_registry",
]
