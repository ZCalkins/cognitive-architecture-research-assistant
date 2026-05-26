"""Seed specialists — real LLMBacking-backed assessment schemas (Session 4).

Strange-loop role
-----------------
The six v0 specialists are object-level oracles (the (C) firewall): each is an
``LLMBacking`` ``Schema`` that, given a paper, returns a single assessment score
plus a Dirichlet confidence via the structured-output protocol. The
orchestrator selects and invokes them; they never decide.

Each specialist has ``latent_dim=1`` (output = [score in 0..1], alpha =
[confidence]). Its ``context_signature`` is sized to the routing embedding
dimension so ``Schema.activate_in`` stays dimensionally consistent —
``LLMBacking.forward`` ignores the context tensor and renders its prompt from
text ``slot_values`` instead, so the output dim and the routing dim are
decoupled.

``citation_graph_position`` and ``author_history`` need data v0 does not have
(no citation graph / publication history). They are real LLM specialists whose
prompts demand an explicitly LOW-confidence assessment from available metadata;
the low alpha records the data scarcity rather than faking signal. Richer
signals arrive with a Semantic-Scholar source (deferred).
"""

from __future__ import annotations

import torch

from src.orchestrator import SchemaRegistry
from src.schemas import LLMBacking, Schema, TypedSlot

SPECIALIST_NAMES = (
    "novelty_vs_kb",
    "methodological_rigor",
    "theoretical_claim_evaluator",
    "relevance_to_projects",
    "citation_graph_position",
    "author_history",
)

_SCORE_INSTRUCTION = (
    "Set the output field to a single float in [0, 1] (your score; higher = "
    "stronger) and the alpha field to a single positive float (your confidence; "
    "higher = more confident)."
)

# referent -> (prompt_template, [(slot_name, type_hint, description), ...])
_SPECIALISTS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "novelty_vs_kb": (
        "You are the novelty-vs-known-work specialist. Assess how novel this "
        "paper is relative to work already in the knowledge base.\n\n"
        "Title: {title}\nAbstract: {abstract}\n\n"
        "Most similar papers already known:\n{kb_neighbors}\n\n"
        "Higher score = more novel relative to what is already known. "
        + _SCORE_INSTRUCTION,
        [
            ("title", "str", "Paper title."),
            ("abstract", "str", "Paper abstract."),
            ("kb_neighbors", "str", "Titles of the most similar known papers."),
        ],
    ),
    "methodological_rigor": (
        "You are the methodological-rigor specialist. From the abstract alone, "
        "assess the apparent methodological rigor (controls, baselines, "
        "ablations, soundness of evaluation).\n\n"
        "Title: {title}\nAbstract: {abstract}\n\n"
        "Higher score = more rigorous. " + _SCORE_INSTRUCTION,
        [
            ("title", "str", "Paper title."),
            ("abstract", "str", "Paper abstract."),
        ],
    ),
    "theoretical_claim_evaluator": (
        "You are the theoretical-claim evaluator. From the abstract, assess "
        "whether the theoretical claims are well-scoped and adequately "
        "supported, as opposed to over-claimed.\n\n"
        "Title: {title}\nAbstract: {abstract}\n\n"
        "Higher score = better-supported claims. " + _SCORE_INSTRUCTION,
        [
            ("title", "str", "Paper title."),
            ("abstract", "str", "Paper abstract."),
        ],
    ),
    "relevance_to_projects": (
        "You are the relevance-to-projects specialist. Assess how relevant this "
        "paper is to the user's active projects.\n\n"
        "Active projects:\n{projects}\n\n"
        "Title: {title}\nAbstract: {abstract}\n\n"
        "Higher score = more relevant. " + _SCORE_INSTRUCTION,
        [
            ("title", "str", "Paper title."),
            ("abstract", "str", "Paper abstract."),
            ("projects", "str", "Descriptions of the user's active projects."),
        ],
    ),
    "citation_graph_position": (
        "You are the citation-graph-position specialist. No citation graph is "
        "available in v0 — assess ONLY from the title, authors, and categories, "
        "and report LOW confidence (a small alpha) to reflect the missing data.\n\n"
        "Title: {title}\nAuthors: {authors}\nCategories: {categories}\n\n"
        "Higher score = likely more central in the citation graph. "
        + _SCORE_INSTRUCTION,
        [
            ("title", "str", "Paper title."),
            ("authors", "str", "Author names."),
            ("categories", "str", "arxiv categories."),
        ],
    ),
    "author_history": (
        "You are the author-history specialist. No publication history is "
        "available in v0 — assess ONLY from the author names, and report LOW "
        "confidence (a small alpha) to reflect the missing data.\n\n"
        "Authors: {authors}\n\n"
        "Higher score = authors likely to produce high-impact work. "
        + _SCORE_INSTRUCTION,
        [
            ("authors", "str", "Author names."),
        ],
    ),
}


def build_specialist(
    referent: str, config: dict, temperature: float = 0.2
) -> Schema:
    """Build one LLMBacking specialist, resolving its model from config."""
    prompt_template, slot_specs = _SPECIALISTS[referent]
    model = config["specialist_models"][config["specialist_assignments"][referent]]
    signature_dim = config["embedding"]["dimension"]
    slots = {
        name: TypedSlot(name, type_hint, description)
        for name, type_hint, description in slot_specs
    }
    backing = LLMBacking(
        prompt_template=prompt_template,
        model=model,
        default_temperature=temperature,
        latent_dim=1,
    )
    return Schema(
        referent=referent,
        backing=backing,
        latent_dim=1,
        slots=slots,
        context_signature=torch.zeros(signature_dim),
    )


def build_v0_specialist_registry(config: dict) -> SchemaRegistry:
    """Wire the six real specialists into a registry for the orchestrator."""
    registry = SchemaRegistry()
    for name in SPECIALIST_NAMES:
        registry.register(build_specialist(name, config))
    return registry
