# Strange Loop Architecture

A research codebase for instantiating and measuring a Hofstadter-style strange
loop in a learnable computational system, organized around a personal research
companion that co-evolves with its user.

**Status:** Phase 0 scaffold + Phase 1 Session 1 complete (polymorphic schema
framework, LLM provider abstraction, orchestrator skeleton, SQLite persistence,
instrumentation, passing tests).

## Quick orientation

- **What this is:** a research vehicle to determine empirically whether an
  orchestrator architecture organized around self-modeling and downward
  causation meets the functional criteria for "I"-hood, and whether the
  combinatorial schema dynamics produce structurally novel cognition on the
  designated frontier domain (AI architecture design).
- **What this isn't:** a portfolio piece, a product, or a claim about
  phenomenal consciousness. See `ARCHITECTURE.md` for the honest scope.
- **Task surface:** v0 paper-triage on AI architecture papers, progressing
  through v1 lit-review drafting, v2 longitudinal project shadow, v3 general
  research-day agent. Each version gated on the previous Phase 5 validation.

## Documents

- `ARCHITECTURE.md` — full project plan: architectural commitments, phase
  progression, Phase 5 verification battery, scaling and risk analysis.
- `CLAUDE.md` — context for Claude Code sessions. Load before opening a
  session.

## Setup

```
uv sync --extra dev
uv run pytest
```

## Project layout

```
src/
    schemas/         Polymorphic Schema framework (types, backings, base)
    providers/       LLMProvider abstraction (Anthropic, Ollama)
    orchestrator/    Registry, workspace, dispatcher, loop (skeleton)
    persistence/     SQLite store for schemas, lifecycle, calibration
    instrumentation/ Probes, ablation harnesses, metrics
    self/            Self-schema, hypernetwork (Phase 3 deliverable)
    inference/       Active inference (Session 2 deliverable)
    substrate/       JEPA latents inside schemas (later sessions)
    memory/          Episodic, semantic, self-narrative tracks (Phase 4)
    dynamics/        Mamba/S6 self-thread (Phase 4)
tests/               Mirrors src/ layout
conf/                Hydra configs
evals/               Phase 5 verification battery (later phases)
experiments/         Hydra-launched experiment scripts
```

## License

MIT.
