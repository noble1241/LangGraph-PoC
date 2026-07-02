# Parallel Multi-Researcher Report Agent — Design Spec

**Date:** 2026-07-02
**Status:** Approved for planning

## Purpose

A CLI research agent built on LangGraph. The user supplies a topic and a number
of researchers. The system generates N researcher personas, pauses for human
approval, then fans out N parallel research subgraphs. Each researcher
iteratively questions the web (Tavily) and Wikipedia, and the collected
research is synthesized into a final report whose sections are written in
parallel. The report prints to the terminal.

## Environment

- **Location:** `C:\Users\noble\Documents\LangGraph-PoC\`
- **Conda env:** `langgraph-poc`, Python 3.11, defined in `environment.yml`
- **LLM:** OpenAI `gpt-4o-mini` via `langchain-openai`
- **Search:** Tavily via `langchain-tavily`; Wikipedia via LangChain's
  `WikipediaRetriever` (`langchain-community` + `wikipedia`, no API key)
- **Secrets:** `.env` file loaded with `python-dotenv`; requires
  `OPENAI_API_KEY` and `TAVILY_API_KEY`. `.env` is git-ignored;
  `.env.example` is committed.

## CLI

```
python research.py "impact of solid-state batteries on EVs" --researchers 3 [--max-turns 2] [--model gpt-4o-mini]
```

- `topic` (positional, required)
- `--researchers` (int, default 3)
- `--max-turns` (int, default 2) — Q&A rounds per researcher
- `--model` (str, default `gpt-4o-mini`)
- Missing args → argparse usage message. Missing API keys → clear fail-fast
  error naming the missing key.

## Architecture

### Parent graph (`ResearchState`)

```
START
  → create_researchers        # LLM structured output → N personas
  → human_approval            # interrupt(); resume with approval or feedback
  → (conditional) ──feedback──→ create_researchers   # regenerate with feedback
  → (approved) Send() fan-out → research subgraph × N (parallel)
  → write_intro ┐
  → write_body  ├ (parallel fan-out after all research collected)
  → write_conclusion ┘
  → finalize_report            # stitch sections + sources list
  → END
```

- Uses a checkpointer (`MemorySaver`) so `interrupt()` works.
- `human_approval` calls `interrupt()`; the CLI prints personas and prompts:
  Enter = approve, any text = feedback. Feedback loops back to
  `create_researchers` (regenerate incorporating feedback), then pauses again.
- `Send()` API dispatches one subgraph instance per approved researcher.

### Research subgraph (`InterviewState`, one per researcher)

```
ask_question
  → search_web (Tavily)      ┐ (parallel fan-out)
  → search_wikipedia         ┘
  → generate_answer
  → (conditional) ──turn_count < max_turns──→ ask_question
  → (else) save_research      # persona-attributed summary + cited sources
  → exit subgraph → parent's completed_research (appended via reducer)
```

- Each search node generates its own `SearchQuery` via structured output from
  the conversation so far, then retrieves.
- `save_research` produces a `CompletedResearch` summary with source URLs.

## Pydantic models (`models.py`)

All domain objects and both graph states are Pydantic `BaseModel`s.

**Domain models** (used with `.with_structured_output()`):

- `Researcher` — `name: str`, `role: str`, `focus: str`;
  `persona` property renders the system-prompt string.
- `ResearcherTeam` — `researchers: list[Researcher]`.
- `SearchQuery` — `query: str`.
- `CompletedResearch` — `researcher: Researcher`, `summary: str`,
  `sources: list[str]`.

**State models** (LangGraph Pydantic state schemas; reducer annotations on
fields):

- `ResearchState` — `topic: str`, `num_researchers: int`, `max_turns: int`,
  `researchers: list[Researcher]`, `human_feedback: str | None`,
  `completed_research: Annotated[list[CompletedResearch], operator.add]`,
  `intro: str`, `body: str`, `conclusion: str`, `final_report: str`.
- `InterviewState` — `messages: Annotated[list, add_messages]`,
  `researcher: Researcher`, `turn_count: int`, `max_turns: int`,
  `sources: Annotated[list[str], operator.add]`,
  `completed_research: Annotated[list[CompletedResearch], operator.add]`
  (shared key so subgraph results merge into the parent).

## Files

| File | Purpose |
|---|---|
| `research.py` | CLI entry: args, key checks, runs graph with checkpointer, approval prompt loop, prints report |
| `agent.py` | `build_graph()` — parent graph + subgraph, all nodes and edges |
| `models.py` | All Pydantic models above |
| `environment.yml` | Conda env spec |
| `.env.example`, `.gitignore`, `README.md` | Config template, ignores, setup/run docs |
| `test_agent.py` | Structure tests with mocked LLM — no keys or network |

## Data flow

topic + N → personas (structured output) → human approval → N parallel
interview loops, each: question → parallel web+wiki retrieval → answer →
loop (≤ max_turns) → `CompletedResearch` → parent state accumulates all N →
intro/body/conclusion written in parallel from all research → finalize →
report string printed to terminal.

## Error handling

- Fail fast on missing API keys before building the graph.
- `recursion_limit` on invocation plus `max_turns` cap per researcher.
- Search node failures (network/API errors) are caught and return an empty
  "no results" context document instead of crashing the parallel batch.

## Testing

- `test_agent.py`: builds both graphs with a mocked LLM, asserts node/edge
  structure, verifies conditional routing (feedback loop, turn-count exit),
  verifies Pydantic state validation rejects bad updates.
- Manual smoke test documented in README:
  `python research.py "test topic" --researchers 2 --max-turns 1`.
