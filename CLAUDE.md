# Claude Code Context — Strange Loop Architecture

> Project memory for Claude Code sessions. Load this first.

## What this project is

A research codebase that instantiates and measures a Hofstadter-style strange
loop as a personal research companion. The strange-loop research subject is
the **orchestrator** — a multi-agent meta-coordinator with a learned
self-model, downward-causal hypernetwork modulation, Piagetian schema
lifecycle, and a persistent self-thread.

LLMs are object-level substrate, invoked as polymorphic schemas. The
orchestrator is the research subject; the LLMs are not.

**Goals:** papers, blog posts, demonstrations. Also: a research companion that
actually serves day-to-day work. The two goals are intentionally coupled; the
architecture is designed to make them mutually reinforcing.

## Honest scope

This builds the **functional** strange loop Hofstadter formalized:
self-modeling, downward causation, recursive self-reference, persistent
self-state, goal-directedness. It does **not** claim to settle phenomenal
consciousness. Phase 5 verification tests the functional claim. Nothing more
is honestly claimable from code.

## Current phase

**Phase 1 — Orchestrator skeleton, Session 2 complete.** LLM call path wired
(structured-output protocol, parse + validate, one corrective retry); active
inference action selection (GenerativeModel, Preferences,
ActiveInferenceSelector) with the top-k baseline preserved as the Phase 5
criterion 2 ablation control; Piagetian lifecycle API scaffolded (compose and
prune wired; spawn/split/merge deferred to Phase 2). Session 1 deliverables
(schemas, providers, orchestrator skeleton, SQLite persistence, probes)
remain. Passing pytest.

**Next session work:** paper ingestion pipeline; ResearchKB integration;
engagement signal handlers (polymorphic event types). Then the six seed
specialists (Session 4-5). Lifecycle trigger *dynamics* land in Phase 2.

## The thesis (one sentence)

The first integrated architecture that operationalizes the Hofstadter strange
loop as five falsifiable functional criteria on a real task surface
(paper-triage on AI architecture papers), with measured effect sizes under
ablation, and tested for combinatorial novelty against the designated
frontier domain.

## Design principles

1. **Falsifiability over plausibility.** Every architectural claim has a Phase
   5 test that could refute it. If you can't write the test, you can't make
   the claim.
2. **Instrumentation is not optional.** Every module exposes probes for
   ablation and measurement. If you can't measure it, you can't claim it.
3. **LLMs are object-level substrate.** The strange-loop research subject is
   the orchestrator, never the LLMs. Object-level work may be delegated to
   LLMs; self-modeling and downward causation may not.
4. **The (C) architectural firewall.** The orchestrator's self-schema is a
   *learned representation causally generative of behavior*, not
   `system_prompt + chat_history`. If the self-schema reduces to the latter,
   the project has failed.
5. **The self-schema is a schema.** Same base class as object schemas.
   Specialness comes from referent and lifecycle position, not from a
   different type. The meta-self-schema is the self-schema taking itself as
   input — no special-casing.
6. **Downward causation is computational, not metaphorical.** It happens via
   hypernetwork modulation with typed output per schema backing (weights /
   prompt-prefix+temperature+model / rule-weights / subgraph routing). If the
   self-model isn't *causally generating* parameters or invocations of its
   substrate, the loop isn't closed.
7. **The substitution.** Project ambitions live in the *system's structural
   capacities* (combinatorial schema space, recursive depth, downward
   causation, schema lifecycle). They do *not* live in the project claims.
   The claim is: this capacity, falsifiably tested on AI architecture design
   as the designated frontier domain, produces or fails to produce measurable
   novelty. Grand framings are long-tail aspirations explicitly bracketed off
   from Phase 5 requirements.
8. **v0 is paper-triage on AI architecture papers.** Generalization happens
   only after Phase 5 validates v0. Progression: v0 → v1 (lit-review
   drafting) → v2 (longitudinal project shadow) → v3 (general research-day
   agent). Premature generalization is the explicit failure mode.
9. **The hard problem stays hard.** No claims about phenomenal experience.
   Stick to functional metrics. No "consciousness" language anywhere.
10. **Version-transition hard gates.** No development of v(n+1) until v(n)
    Phase 5 passes with documented effect sizes.
11. **Safety boundary at v4 and beyond.** The system may modify its schema
    population. It may not modify its own framework code. Non-negotiable.

## Don'ts

- **Don't conflate logging with self-modeling.** A logfile of past states is
  not a self-schema. A self-schema *predicts*, *generates uncertainty*, and
  *modulates*.
- **Don't drift toward Layer-1 supervisor pattern.** Meta-decisions stay in
  the orchestrator. Specialists are oracles, never deciders.
- **Don't skip ablations.**
- **Don't optimize before measuring.**
- **Don't add dependencies casually.** F1 framework choice means we own our
  orchestrator; importing LangChain / LangGraph / AutoGen / CrewAI breaks the
  (C) firewall.
- **Don't anthropomorphize in code or docs.** Functional language only.
- **Don't propose framework-code self-modification as a feature.**
- **Don't generalize the task surface ahead of Phase 5.**

## Stack

- Python 3.11+, PyTorch 2.4+
- `uv` for packaging
- Hydra for configs
- pytest + Hypothesis for testing
- ruff for lint/format
- Custom `LLMProvider` wrapping Anthropic SDK + Ollama
- SQLite for v0 state persistence
- **No** LangChain, LangGraph, AutoGen, CrewAI, SQLAlchemy

## Repo conventions

- Public API defined in `__init__.py` of each subpackage
- Type hints required on all public functions
- Docstrings: NumPy style. Core classes include a `Strange-loop role` section.
- Tests mirror `src/` layout
- Every test names which (A/B/C/D) claim or Phase 5 criterion it serves

## Working with this codebase

When a session starts, read this file, the relevant section of
`ARCHITECTURE.md` for the current phase, and the module-level docstrings of
any file you're modifying.

For each task, identify:
1. Which phase it belongs to.
2. Which strange-loop criterion or supporting claim it serves.
3. What ablation or test will verify it works.

If a task can't answer (2) and (3), pause and surface that.

## Glossary

- **Schema** — a parameterized module representing some referent. Polymorphic
  across `NeuralBacking`, `LLMBacking`, `SymbolicBacking`, `CompositeBacking`.
- **Self-schema** — the schema whose referent is the orchestrator's own
  state. Phase 3 deliverable.
- **Orchestrator** — the multi-agent meta-coordinator. The strange-loop
  research subject.
- **Hypernetwork** — module taking self-schema state as input and producing
  typed modulations per schema backing. Phase 3 deliverable.
- **Downward causation** — high-level (self-schema) state causally generating
  low-level (object-schema) parameters or invocations via the hypernetwork.
- **Strange loop** — a tangled hierarchy where levels feed back. Here: the
  self-schema modulating the substrate that produced it.
- **Functional I** — the system meets all five Phase 5 criteria.
- **(C) architectural firewall** — the principle that the orchestrator (not
  the LLMs) is the strange-loop research subject.
- **Piagetian lifecycle** — spawn/split/merge/prune dynamics. Phase 2.
- **Self-thread** — Mamba/S6 state-space model ingesting self-schema states.
  Phase 4.
- **Identity invariant** — low-dimensional projection of the self-thread,
  regularized to remain stable across sessions and substrate changes.
- **Designated frontier domain** — AI architecture design.
- **The substitution** — grand ambitions live in structural capacities, not
  project claims.
- **v0 / v1 / v2 / v3** — paper-triage / lit-review drafting / longitudinal
  project shadow / general research-day agent.
