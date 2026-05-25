"""Seed specialists — STUBS in Session 3, real LLMBacking-backed in Session 4.

See ARCHITECTURE.md "Phase 1 deliverable 3". These stubs only wire the
end-to-end pipeline; they are not load-bearing for any architectural claim.
"""

from src.specialists.stubs import (
    build_author_history_stub,
    build_citation_graph_position_stub,
    build_methodological_rigor_stub,
    build_novelty_vs_kb_stub,
    build_relevance_to_projects_stub,
    build_theoretical_claim_evaluator_stub,
    build_v0_specialist_registry,
)

__all__ = [
    "build_author_history_stub",
    "build_citation_graph_position_stub",
    "build_methodological_rigor_stub",
    "build_novelty_vs_kb_stub",
    "build_relevance_to_projects_stub",
    "build_theoretical_claim_evaluator_stub",
    "build_v0_specialist_registry",
]
