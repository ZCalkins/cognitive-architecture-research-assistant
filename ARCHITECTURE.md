# Architecture — Strange Loop

## Project goal

Instantiate, measure, and study a Hofstadter-style strange loop in a learnable
computational system, organized around a personal research companion that
co-evolves with its user. Determine empirically whether the resulting
orchestrator architecture satisfies the functional criteria for "I"-hood, and
whether the combinatorial schema dynamics produce structurally novel cognition
on a designated frontier domain.

## Honest scope

Three things to keep separated at all times:

1. **What this builds:** a multi-agent orchestrator whose self-model is causally
   generative of its routing, prompt-construction, and specialist-selection
   behavior; whose schema population dynamically reorganizes via Piagetian
   lifecycle; whose persistent self-thread maintains identity invariants across
   sessions and substrate changes.
2. **What this can claim, if it works:** the functional strange loop Hofstadter
   formalized — self-modeling, downward causation, recursive self-reference,
   persistent self-state, goal-directedness — instantiated at the orchestrator
   level and falsifiably tested by the Phase 5 verification battery. Also:
   structurally novel combinatorial cognition on the designated frontier domain
   (AI architecture design), where "novel" means producing framings expert
   evaluators find non-trivial.
3. **What this cannot prove:** that any of the above gives rise to phenomenal
   experience. The hard problem stays hard. No claims about consciousness,
   qualia, or what-it's-like-ness anywhere in the codebase or docs.

The Phase 5 verification battery rigorously tests the functional claim.
Combinatorial novelty is tested separately against the designated frontier
domain. Failure on either is a valid result, not a setback.

## The thesis

**Primary claim (D):** the first integrated architecture that operationalizes
the Hofstadter strange loop as five falsifiable functional criteria on a real
task surface, demonstrated with measured effect sizes and ablation studies.

**Supporting mechanisms:**

- **(A) co-evolutionary identity** — the orchestrator's self-model, schema
  population, calibrated capabilities, and persistent character co-evolve with
  the user, producing measurable structural personalization no LLM-wrapper
  achieves.
- **(B) computational downward causation** — the self-schema's state causally
  determines object-level behavior via hypernetwork-generated modulations of
  schema parameters, prompt prefixes, model selection, and routing.
- **(C) self-organizing specialist emergence** — specialists are not designed
  in advance; they emerge from Piagetian lifecycle dynamics over real workload
  prediction errors.

**Combinatorial novelty test:** the architecture's schema network supports
combinatorial generation of cognitive content not present in any individual
schema's pretraining, measured against the designated frontier domain (AI
architecture design) and evaluated by expert review.

## Architectural commitments

| Component | Source | Strange-loop role |
|---|---|---|
| Polymorphic schema network with Piagetian lifecycle | Piaget; modern modular ML | Dynamic structure; schemas spawn/split/merge/prune; the self-schema is one schema among many |
| Self-schema + hypernetwork modulation | Ha et al. | **Mechanism of downward causation** — self-model state generates typed modulations per backing type |
| Active Inference / FEP | Friston | Unified prediction + action selection at the orchestrator level |
| Global workspace (context-construction service) | Baars; Dehaene | Assembles per-invocation context from active schemas, self-schema, KB retrievals |
| EDL + Subjective Logic | Sensoy; Jøsang | Principled self-uncertainty via Dirichlet-parameterized beliefs |
| Mamba/S6 self-thread | Gu, Dao | Persistent temporal subject; identity invariants across thousands of sessions |
| JEPA-style latents (within schemas, not as substrate) | LeCun | Learned representations of text, trajectories, agent execution traces inside specific schemas |
| Test-time online learning | TTRL; MAML family | Continual self-modification of self-schema and schema population |
| LLM substrate (frozen, as LLMBacking) | various vendors | Object-level cognitive substrate; explicitly not the research subject |

The integration is the contribution. None of these alone is novel. The
combination, organized around an orchestrator-first strange-loop architecture
and operationalized as five falsifiable Phase 5 criteria, is.

## Architecture overview

The system is a multi-agent orchestrator whose internal state is the
strange-loop research subject. LLMs are substrate-of-action, invoked as
object-level schemas. The orchestrator's self-schema models the schema
population and modulates it via the hypernetwork.

**Heterogeneous substrate:**

- Frozen LLMs (as LLMBacking inside polymorphic schemas)
- Learned-from-scratch neural modules (self-schema, hypernetwork, Mamba
  self-thread, some specialist schemas)
- ResearchKB (semantic-memory substrate, integrated by reference)
- JEPA-style latents over agent execution traces and paper representations
  (inside specific schemas)

**Cognitive division of labor (the (C) firewall):**

- LLMs do object-level work: summarization, novelty assessment, drafting,
  retrieval, code analysis.
- The orchestrator does meta-cognition: which specialist to invoke, what
  context to assemble, when to spawn schemas, when to escalate, when to update
  the self-schema.
- The self-schema is never `system_prompt + chat_history`. It is a learned
  representation causally generative of behavior. If implemented as the
  former, the project has failed.

## Task surface

**v0 — paper triage on AI architecture papers.** Every batch cycle, the system
ingests new items from arxiv/Semantic-Scholar in AI architecture (broadly:
cognitive architectures, agent systems, world models, reasoning systems, novel
substrate designs), produces a ranked "you should read these" list with
structured novelty assessments against ResearchKB, and learns from engagement
signals (read/skim/discard/flag/time-on-paper/notes).

**v1 — lit-review drafting on demand.** Composes v0 specialists into structured
draft production for named topics.

**v2 — longitudinal project shadow.** Tracks designated projects (SSN,
strange-loop itself, ResearchKB), maintains a model of decisions/blockers/
avoided-questions, surfaces patterns.

**v3 — general research-day agent.** Plans the research day across all task
domains. Composes v0/v1/v2 capabilities with cross-domain synthesis and
meta-cognitive support.

**Hard gates:** each version requires Phase 5 validation on the previous before
development of the next begins. Premature generalization is the explicit
failure mode.

## Phases

### Phase 0 — Scaffold and synthesis

**Status:** completed in the from-scratch build. Package structure;
polymorphic Schema framework; LLM provider abstraction; orchestrator
skeleton (registry, workspace, dispatcher, loop); SQLite persistence;
probe instrumentation; this document and `CLAUDE.md`; passing pytest.

### Phase 1 — Orchestrator skeleton

**Status:** Session 1 completed in the from-scratch build (polymorphic
schemas, LLM provider, orchestrator skeleton, persistence). Remaining
sessions: active inference action selection (Session 2), paper ingestion
and ResearchKB integration (Session 3), engagement signal pipeline
(Session 3), six seed specialists (Session 4-5).

**Goal:** trivial end-to-end paper-triage running with hand-designed
specialists. No self-schema yet, no hypernetwork, no lifecycle.

**Deliverables:**

1. Polymorphic schema framework (full interface: slots, parents/children,
   provenance, context-signature, polymorphic backings). **DONE in Session 1.**
2. `LLMProvider` abstraction wrapping Anthropic SDK (primary) and Ollama
   (secondary). **DONE in Session 1.**
3. ~6 hand-designed seed specialists with explicit slots and confidence
   interfaces:
   - novelty-vs-KB
   - methodological-rigor
   - theoretical-claim-evaluator
   - relevance-to-Zach's-projects
   - citation-graph-position
   - author-history
4. ResearchKB integration (read access; write access for newly-ingested
   papers).
5. Paper ingestion pipeline (daily arxiv/Semantic-Scholar crawl with change
   detection).
6. Engagement signal pipeline (polymorphic event types; v0 handlers for
   read/skim/discard/flag/notes/time-on-paper).
7. Global workspace context-construction service. **DONE in Session 1.**
8. Active inference action selection over specialist invocations.
9. SQLite-backed state persistence. **DONE in Session 1.**

**Milestone test:** system produces a daily triage; engagement signals are
captured; baseline external observer (Claude with full logs) can be compared
on next-paper prediction. No claim yet about self-modeling — this is the
substrate.

### Phase 2 — Schema lifecycle

**Goal:** system spawns its own specialists in response to engagement-signal
prediction error.

**Deliverables:**

1. Lifecycle trigger definitions:
   - **Spawn:** N consistent prediction errors in a localized region of input
     space, with buffering against single anomalies. Rate-limited.
   - **Split:** a single schema's predictions are bimodal over a sustained
     window.
   - **Merge:** two schemas exhibit high mutual prediction similarity over a
     sustained window.
   - **Prune:** extended low activation + low utility (no recent positive
     engagement signal).
2. Adaptive thresholds (functions of current population size and error-rate
   distribution).
3. Provenance logging for every lifecycle event.

**Milestone test:** introduce a new paper subtype the seed specialists handle
poorly → system spawns a new specialist → engagement-signal prediction error
in that region drops.

### Phase 3 — Self-schema and hypernetwork

**Goal:** self-schema modulates routing, context construction, and specialist
invocation. Phase 5 criterion 2 (downward causation traceability) becomes
testable.

**Deliverables:**

1. Self-schema definition. Inputs: active schema set, recent per-schema
   prediction errors, recent workspace broadcasts, recent specialist
   invocations, current free-energy estimate, recent self-schema states.
2. EDL/Subjective Logic outputs: Dirichlet-parameterized self-uncertainty.
3. Hypernetwork with typed output per schema backing:
   - **NeuralBacking:** weight modulations
   - **LLMBacking:** prompt-prefix + temperature + model-selection modulations
   - **SymbolicBacking:** rule-weighting / parameter modulations
   - **CompositeBacking:** subgraph routing weights
4. Self-schema as privileged broadcaster in the workspace (its state is always
   part of assembled context).
5. Meta-self-schema (self-schema taking itself as input).

**Milestone tests:**

- Ablate self-schema components → specialist-routing distribution shifts in
  predicted ways.
- Self-schema predicts orchestrator's next state better than same-capacity
  external observer with identical observables.
- Meta-self-schema produces measurably different content from level-1.

### Phase 4 — Self-thread and identity invariants

**Goal:** persistent temporal subject across thousands of triage sessions.

**Deliverables:**

1. Mamba/S6 self-thread ingesting self-schema states; produces compressed
   running representation.
2. Self-narrative episodic memory indexed by salience.
3. Identity invariants: low-dimensional projections regularized to stay stable
   across sessions and substrate changes.
4. Checkpointing every batch + every session close; cold-start resume
   preserves self-thread continuity.

**Open architectural question (flagged honestly):** identity invariants may
need to be partitioned per task-domain rather than global. Module designed to
support either; empirically determined at v3 transition.

**Milestone test:** across thousands of triage events, self-thread maintains
coherent identity invariants while adapting to engagement-signal feedback.
Perturbations (LLM backend swap, schema-population restart, noise injection)
produce recovery trajectories that respect invariants.

### Phase 5 — Verification battery

Re-instantiated for the paper-triage task surface. All five criteria must pass
with effect sizes that survive ablation. A failed criterion is informative —
it tells you which component is underbuilt.

1. **Self-prediction superiority.** The self-schema predicts orchestrator
   next-state on Zach's paper-engagement behavior better than an external
   Claude/GPT instance with full identical logs. Measured over a held-out
   paper window.
2. **Downward causation traceability.** Ablate or noise self-schema components
   → predicted shifts in specialist-routing distribution. Effect sizes must
   match advance predictions.
3. **Recursive depth.** Meta-novelty assessment ("what kind of paper is this
   system bad at assessing") produces non-trivial new content rather than
   refinements of level-1. Measured by held-out expert evaluation.
4. **Counterfactual self-knowledge.** Orchestrator predicts its own behavior
   on hypothetical paper batches with calibrated accuracy. Measured against
   actual subsequent behavior on similar batches.
5. **Behavioral coherence under perturbation.** Swap underlying LLM backend,
   restart schema population cold, inject specialist-output noise → identity
   invariants in self-thread recover. The system still feels like itself in
   measurable ways.

**Combinatorial novelty test (separate from the five-criterion battery):**
after N months of operation, present the system's accumulated schema
combinations and outputs on AI architecture topics to expert evaluators (Zach
+ peer network). Measure novelty/coherence/usefulness against baselines
(Claude with full ResearchKB access). Result categories: trivially
recombinatorial, incoherent, wrong-but-interesting (productive failure),
substantively novel, paradigm-shifting (vanishingly unlikely). Each is a valid
result.

### Phase 6+ — v1 and beyond

Gated on Phase 5 validation. v1 adds lit-review drafting. v2 adds longitudinal
project shadow. v3 adds general research-day. Each requires its own Phase 5
re-instantiation as a hard gate.

## Polymorphic schema design

Each schema is a `(metadata, content, interface)` triple.

**Metadata:**

- `referent: str`
- `slots: dict[str, TypedSlot]` — Minsky-frame slots, downstream-referenceable
- `parents: list[Schema]` — generalization hierarchy
- `children: list[Schema]` — specialization hierarchy
- `provenance: SpawnRecord` — who/what/when spawned this, in response to what
  error
- `confidence_trajectory: list[float]`
- `context_signature: Tensor` — what contexts activate this schema

**Content:**

- `embedding: Tensor` — latent semantic content
- `backing: SchemaBacking` — polymorphic implementation:
  - `NeuralBacking` — learned-from-scratch neural module
  - `LLMBacking` — prompt template + model selection
  - `SymbolicBacking` — rule or program
  - `CompositeBacking` — graph over other schemas

**Interface:**

- `predict(context, slot_values) -> (output, dirichlet_alpha)`
- `map_to(target_schema) -> StructureMapping` — analogical mapping (Gentner SMT)
- `specialize(constraint) -> Schema`
- `generalize() -> Schema`
- `compose(other, mode) -> Schema`
- `activate_in(context) -> float` — context-relevance scoring
- `ablate()`, `restore()` — for Phase 5

**Load-bearing human-cognition features (kept):**

- Slot/role structure (Minsky frames) — substrate for hypernetwork modulation
- Analogical structure-mapping (Gentner SMT) — mechanism for combinatorial
  novelty
- Generalization/specialization operators — Piagetian lifecycle given
  algebraic structure
- Context-sensitive activation — substrate for (A) personalization
- Composability (schemas containing schemas) — makes the schema factory
  generative
- Inspectable provenance — required by (D) falsifiability

**Human features explicitly rejected (kept out):**

- Working memory limits
- Forgetting curves / decay
- Confirmation bias and motivated reasoning
- Affective coloring of schemas
- Embodied/grounded semantics (Barsalou simulators)
- Prototype-only or exemplar-only representation (modern hybrid subsumes both)

## Memory architecture

Five tracks. Each declares what's persisted to disk, what's in working memory,
what's recomputable.

1. **Episodic** — paper-engagement events (paper-id, timestamp, signal type,
   signal payload). Persisted.
2. **Semantic** — ResearchKB (existing) extended with schema-population
   history. Persisted, integrated by reference.
3. **Self-narrative** — Mamba self-thread state and identity invariants.
   Checkpointed every batch + every session close.
4. **Procedural** — schema lifecycle log (spawn/split/merge/prune events with
   full triggers and provenance). Persisted.
5. **Calibration** — per-schema confidence trajectory grounding Dirichlet/EDL
   outputs. Persisted.

## Global workspace

Reframed from "broadcast layer" to **context-construction service**. When a
specialist is invoked, the workspace assembles its prompt-context from:

- Currently active schemas (context-signature match)
- Self-schema state (always included, as text or hypernetwork-mediated
  parameter modulation)
- Recent workspace contents (rolling window)
- KB retrievals (per-specialist, scoped by slot values)

The self-schema is a privileged broadcaster in the sense that its state is
*always* part of every assembled context. Privilege is implemented
architecturally, not declared philosophically.

## Runtime model

**R2 — Scheduled batch + interactive session attach.**

- Daily (or twice-daily) batch job processes new papers, advances orchestrator
  one cycle, updates self-thread, persists state.
- Interactive sessions: Zach attaches, queries/reads/flags, signals captured,
  state persisted on close.
- Always-on for *state*, event-driven for *compute*.
- Mamba self-thread treats "between sessions" as elapsed sequence positions in
  a coherent logical thread; continuous wall-clock running is not required.

**Persistence:** SQLite for v0. Postgres if/when concurrency demands it.
Self-thread Mamba state checkpointed every batch + every session close (the
state is small, persistence is cheap).

**Scheduling:** cron or systemd timer for the batch job. Interactive session
is a script. No Airflow/Prefect at this scale.

## Multi-agent framework

**F1 — Build from scratch with thin vendor-SDK wrappers.**

- Custom orchestrator loop, schema registry, prompt constructor, response
  parser, multi-agent coordination.
- `LLMProvider` interface wrapping vendor SDKs directly:
  - Primary: Anthropic SDK (Claude)
  - Secondary: Ollama (local Llama/Qwen) for Phase 5 criterion 5 swap-test
  - OpenAI later, if at all
- No LangChain, LangGraph, AutoGen, CrewAI as dependencies.
- Rationale: protects the (C) firewall; framework primitives are designed
  *for* the strange-loop architecture, not adapted from existing patterns.
  End-to-end auditability for Phase 5.

## What we are not doing (deliberate exclusions)

- **No LLM-as-the-system.** LLMs are object-level substrate. The strange-loop
  research subject is the orchestrator.
- **No symbolic-only architecture.** Schemas have learned content; the
  architecture is hybrid by design.
- **No claims about phenomenal experience.** Every result reported in
  functional terms.
- **No "consciousness" language anywhere in the docs.** Phenomenal-substrate
  vocabulary has no place here.
- **No Attention Schema Theory module** as an explicit commitment. The
  self-schema already models which schemas are active. Demoted to optional,
  only revisited if Phase 5 leaves a measurable gap AST might close.
- **No Higher-Order Thought layer** as an explicit commitment. Same reasoning.
- **No generative self-narration via LLM-narrator.** Language is everywhere in
  the system; the narrator role is dissolved. Calibration of self-reports
  moves to the orchestrator directly (criterion 4).
- **No MiniGrid, no Crafter.** The environment is the live arxiv stream +
  ResearchKB + engagement signal.

## Scaling and risks

### Planned mitigations (Phase 2 or later)

- **MoE-style hypernetwork output.** As registry grows from ~6 (v0) to ~200
  (v3), hypernetwork output dimension grows linearly. Mitigation: gated/sparse
  output — hypernetwork modulates only the top-k active schemas, not all.
- **Adaptive lifecycle hyperparameters.** Spawn/split/merge/prune thresholds
  tuned for ~6 seeds will misbehave at ~200. Mitigation: thresholds expressed
  as functions of current population size and error-rate distribution.
- **Polymorphic engagement signal pipeline.** v0 signals are paper-engagement;
  later versions add code commits, project decisions, calendar moves.
  Mitigation: event-type registration pattern from v0.
- **Phase 5 re-instantiation at each version transition.** Criteria are not
  inherited; each version writes its own version of the five tests against
  its own task surface.

### Empirical risks (honest open questions)

- **Self-thread compression at scale.** Mamba's compression of self-state
  history into low-dimensional identity invariants is conjecture for
  heterogeneous self-state sequences. May turn out v3 needs partitioned
  invariants per task-domain rather than one global invariant.
- **Self-schema learnability at large registry size.** Training dynamics with
  200+ schemas are not guaranteed stable. v3 may require hierarchical
  self-schemas.
- **Identity-invariant meaningfulness.** Across triage + drafting + project
  shadow + planning, "identity" may not be a coherent invariant. If so,
  that's a Phase 5 criterion 5 failure and a valid result.

### Disciplinary risks (require active maintenance)

- **(C) firewall under pressure.** As specialists become more capable,
  temptation to delegate meta-cognition to them grows. Discipline: every
  meta-decision stays in the orchestrator. Specialists are oracles, never
  deciders.
- **Version-transition premature generalization.** Each version will pressure
  skipping the previous Phase 5. Hard rule: no v(n+1) development until v(n)
  Phase 5 passes with documented effect sizes.

### Safety boundary (non-negotiable)

- **No framework-code self-modification.** The system can modify its schema
  population. It cannot modify its own framework code. Architectural
  self-improvement, if pursued at v4, takes the form of the system
  *proposing changes for human review*, never *executing changes to itself*.
  Phase 5 success criteria never include "system improves its own framework."

## References (curated)

- Hofstadter, *I Am a Strange Loop* (2007)
- Friston et al., active inference papers (2010s–present)
- Baars, *A Cognitive Theory of Consciousness* (1988); Dehaene,
  *Consciousness and the Brain* (2014)
- Ha, Dai, Le, "HyperNetworks" (2016)
- Sensoy, Kaplan, Kandemir, "Evidential Deep Learning to Quantify
  Classification Uncertainty" (2018)
- Jøsang, *Subjective Logic* (2016)
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  (2023)
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022); V-JEPA
  papers
- Piaget, *The Construction of Reality in the Child* (1954)
- Gentner, "Structure-Mapping: A Theoretical Framework for Analogy" (1983)
- Minsky, "A Framework for Representing Knowledge" (1974)
