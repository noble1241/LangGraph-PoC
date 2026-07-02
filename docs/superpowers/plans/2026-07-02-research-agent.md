# Parallel Multi-Researcher Report Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI LangGraph app: topic + N researchers in → human-approved personas → N parallel research subgraphs (Tavily + Wikipedia) → parallel-written final report printed to terminal.

**Architecture:** Parent `StateGraph` (create researchers → `interrupt()` approval → `Send()` fan-out → parallel intro/body/conclusion → finalize). Each `Send` runs a compiled interview subgraph: ask_question → parallel web+wiki search → generate_answer → loop until `max_turns` → save_research. All state schemas and structured outputs are Pydantic `BaseModel`s.

**Tech Stack:** Python 3.11 (conda env `langgraph-poc`), langgraph, langchain-openai (`gpt-4o-mini`), langchain-tavily, langchain-community + wikipedia (WikipediaRetriever), python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-research-agent-design.md`. Deviations from spec (approved during planning): `InterviewState` gains `topic: str` and `context: Annotated[list[str], operator.add]` fields (needed by prompts/search nodes); routing functions live at module level for testability.

## Global Constraints

- Project root: `C:\Users\noble\Documents\LangGraph-PoC\` — all paths below are relative to it.
- Conda env is named `langgraph-poc`, Python `3.11`. All run/test commands use `conda run -n langgraph-poc ...` so no shell activation is needed.
- LLM default: `gpt-4o-mini`, `temperature=0`, overridable via `--model`.
- API keys ONLY via `.env` (`OPENAI_API_KEY`, `TAVILY_API_KEY`), loaded with `python-dotenv`. `.env` is git-ignored; `.env.example` is committed. Fail fast with a clear message if a key is missing.
- Every domain object and both graph states are Pydantic `BaseModel`s (no TypedDict).
- Tests must run with NO network and NO API keys (mocked LLM + mocked search tools).
- Defaults: `--researchers 3`, `--max-turns 2`; invocation `recursion_limit` 100.
- Report goes to stdout only (no file output).

---

### Task 1: Project scaffolding + conda environment

**Files:**
- Create: `environment.yml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a working conda env `langgraph-poc` in which every later `conda run -n langgraph-poc` command executes; git-ignored `.env`.

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: langgraph-poc
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - langgraph>=0.6
      - langchain-openai>=0.3
      - langchain-tavily>=0.2
      - langchain-community>=0.3
      - wikipedia>=1.4
      - python-dotenv>=1.0
      - pytest>=8.0
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.env
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Write `.env.example`**

```dotenv
OPENAI_API_KEY=
TAVILY_API_KEY=
```

- [ ] **Step 4: Write `README.md`**

```markdown
# LangGraph-PoC — Parallel Multi-Researcher Report Agent

A LangGraph research agent: give it a topic and a number of researchers.
It generates researcher personas (you approve or give feedback), then each
researcher interviews the web (Tavily) and Wikipedia in parallel, and the
findings are synthesized into a report written to your terminal.

## Setup

```bash
conda env create -f environment.yml
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY and TAVILY_API_KEY
```

## Run

```bash
conda run -n langgraph-poc --no-capture-output python research.py "impact of solid-state batteries on EVs" --researchers 3
```

Options: `--researchers N` (default 3), `--max-turns N` Q&A rounds per
researcher (default 2), `--model NAME` (default gpt-4o-mini).

When the proposed researcher team is shown, press Enter to approve it, or
type feedback (e.g. "replace the economist with a supply-chain expert") to
regenerate.

## Tests (no API keys needed)

```bash
conda run -n langgraph-poc pytest test_agent.py -v
```

## Smoke test (real APIs, cheap)

```bash
conda run -n langgraph-poc --no-capture-output python research.py "history of the espresso machine" --researchers 2 --max-turns 1
```
```

- [ ] **Step 5: Create the conda env**

Run: `conda env create -f environment.yml`
Expected: ends with instructions to activate `langgraph-poc` (takes a few minutes).

- [ ] **Step 6: Verify imports**

Run: `conda run -n langgraph-poc python -c "import langgraph, langchain_openai, langchain_tavily, langchain_community, wikipedia, dotenv, pytest; from langgraph.types import Send, interrupt, Command; from langgraph.checkpoint.memory import InMemorySaver; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add environment.yml .gitignore .env.example README.md
git commit -m "chore: scaffold project, conda env, and docs"
```

---

### Task 2: Pydantic models (`models.py`)

**Files:**
- Create: `models.py`
- Create: `test_agent.py` (models section)

**Interfaces:**
- Consumes: nothing.
- Produces (used verbatim by Tasks 3–5):
  - `Researcher(name: str, role: str, focus: str)` with `.persona -> str` property
  - `ResearcherTeam(researchers: list[Researcher])`
  - `SearchQuery(query: str)`
  - `CompletedResearch(researcher: Researcher, summary: str, sources: list[str])`
  - `ResearchState(topic, num_researchers, max_turns, researchers, human_feedback, completed_research, intro, body, conclusion, final_report)`
  - `InterviewState(messages, researcher, topic, max_turns, turn_count, context, sources, completed_research)`

- [ ] **Step 1: Write the failing tests** — create `test_agent.py`:

```python
"""Tests for the research agent. All tests run offline: LLM and search tools are mocked."""
import pytest
from pydantic import ValidationError

from models import (
    CompletedResearch,
    InterviewState,
    Researcher,
    ResearcherTeam,
    ResearchState,
    SearchQuery,
)

RESEARCHER = Researcher(name="Ada Vale", role="Materials scientist", focus="battery chemistry")


class TestModels:
    def test_researcher_persona_renders_all_fields(self):
        persona = RESEARCHER.persona
        assert "Ada Vale" in persona
        assert "Materials scientist" in persona
        assert "battery chemistry" in persona

    def test_research_state_defaults(self):
        state = ResearchState(topic="fusion")
        assert state.num_researchers == 3
        assert state.max_turns == 2
        assert state.researchers == []
        assert state.completed_research == []
        assert state.final_report == ""

    def test_research_state_requires_topic(self):
        with pytest.raises(ValidationError):
            ResearchState()

    def test_interview_state_requires_researcher(self):
        with pytest.raises(ValidationError):
            InterviewState(topic="fusion")

    def test_completed_research_roundtrip(self):
        cr = CompletedResearch(researcher=RESEARCHER, summary="found things", sources=["http://a"])
        assert CompletedResearch.model_validate(cr.model_dump()) == cr

    def test_structured_output_models_validate(self):
        assert ResearcherTeam(researchers=[RESEARCHER]).researchers[0].name == "Ada Vale"
        assert SearchQuery(query="solid state batteries").query
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: FAIL at import time — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Write `models.py`**

```python
"""Pydantic domain models and LangGraph state schemas."""
import operator
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Researcher(BaseModel):
    """A researcher persona generated by the LLM."""
    name: str = Field(description="Full name of the researcher persona.")
    role: str = Field(description="Professional role, e.g. 'Battery materials scientist'.")
    focus: str = Field(description="One sentence: the specific angle this researcher investigates.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nFocus: {self.focus}"


class ResearcherTeam(BaseModel):
    """Structured output for researcher-team generation."""
    researchers: list[Researcher] = Field(description="The team of researcher personas.")


class SearchQuery(BaseModel):
    """Structured output for search-query generation."""
    query: str = Field(description="A well-formed web search query.")


class CompletedResearch(BaseModel):
    """One researcher's finished, persona-attributed research memo."""
    researcher: Researcher
    summary: str
    sources: list[str]


class ResearchState(BaseModel):
    """Parent graph state."""
    topic: str
    num_researchers: int = 3
    max_turns: int = 2
    researchers: list[Researcher] = []
    human_feedback: str | None = None
    completed_research: Annotated[list[CompletedResearch], operator.add] = []
    intro: str = ""
    body: str = ""
    conclusion: str = ""
    final_report: str = ""


class InterviewState(BaseModel):
    """Research subgraph state (one instance per researcher, run in parallel)."""
    messages: Annotated[list, add_messages] = []
    researcher: Researcher
    topic: str = ""
    max_turns: int = 2
    turn_count: int = 0
    context: Annotated[list[str], operator.add] = []
    sources: Annotated[list[str], operator.add] = []
    # Shares its name with ResearchState so subgraph results merge into the parent.
    completed_research: Annotated[list[CompletedResearch], operator.add] = []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add models.py test_agent.py
git commit -m "feat: add Pydantic domain models and graph state schemas"
```

---

### Task 3: Research (interview) subgraph (`agent.py`, part 1)

**Files:**
- Create: `agent.py`
- Modify: `test_agent.py` (append interview-graph section)

**Interfaces:**
- Consumes: everything in `models.py` (Task 2).
- Produces (used by Task 4):
  - `build_interview_graph(llm, web_search, wiki_retriever) -> CompiledStateGraph`
  - `route_after_answer(state: InterviewState) -> str` (module level, returns `"save_research"` or `"ask_question"`)
  - Fake-injection convention: `llm.invoke(...) -> AIMessage`, `llm.with_structured_output(Schema).invoke(...) -> Schema instance`, `web_search.invoke({"query": str}) -> {"results": [{"url","content","title"}]}`, `wiki_retriever.invoke(str) -> list[Document]`.

- [ ] **Step 1: Append failing tests to `test_agent.py`**

Add imports at the top of the file:

```python
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agent import build_interview_graph, route_after_answer
```

Add fake factories (module level, below `RESEARCHER`) and the test class:

```python
def make_fake_llm():
    """MagicMock LLM: plain .invoke returns an AIMessage; with_structured_output
    returns a mock whose .invoke returns a valid instance of the requested schema."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="fake LLM response")

    def structured(schema, **kwargs):
        m = MagicMock()
        if schema is ResearcherTeam:
            m.invoke.return_value = ResearcherTeam(researchers=[
                Researcher(name="Ada Vale", role="Scientist", focus="chemistry"),
                Researcher(name="Ben Ito", role="Economist", focus="markets"),
            ])
        elif schema is SearchQuery:
            m.invoke.return_value = SearchQuery(query="fake query")
        else:
            raise AssertionError(f"unexpected schema {schema}")
        return m

    llm.with_structured_output.side_effect = structured
    return llm


def make_fake_web_search(fail=False):
    tool = MagicMock()
    if fail:
        tool.invoke.side_effect = RuntimeError("tavily down")
    else:
        tool.invoke.return_value = {
            "results": [{"url": "http://web.example/a", "content": "web fact", "title": "A"}]
        }
    return tool


def make_fake_wiki(fail=False):
    retriever = MagicMock()
    if fail:
        retriever.invoke.side_effect = RuntimeError("wiki down")
    else:
        retriever.invoke.return_value = [
            Document(page_content="wiki fact", metadata={"source": "http://wiki.example/b"})
        ]
    return retriever


class TestInterviewGraph:
    def build(self, **kwargs):
        return build_interview_graph(
            kwargs.get("llm", make_fake_llm()),
            kwargs.get("web_search", make_fake_web_search(fail=kwargs.get("web_fail", False))),
            kwargs.get("wiki_retriever", make_fake_wiki(fail=kwargs.get("wiki_fail", False))),
        )

    def test_graph_has_expected_nodes(self):
        nodes = set(self.build().get_graph().nodes)
        assert {"ask_question", "search_web", "search_wikipedia",
                "generate_answer", "save_research"} <= nodes

    def test_route_after_answer(self):
        going = InterviewState(researcher=RESEARCHER, turn_count=1, max_turns=2)
        done = InterviewState(researcher=RESEARCHER, turn_count=2, max_turns=2)
        assert route_after_answer(going) == "ask_question"
        assert route_after_answer(done) == "save_research"

    def test_full_interview_produces_completed_research(self):
        graph = self.build()
        result = graph.invoke({"researcher": RESEARCHER, "topic": "espresso", "max_turns": 1})
        assert len(result["completed_research"]) == 1
        memo = result["completed_research"][0]
        assert memo.researcher.name == "Ada Vale"
        assert memo.summary == "fake LLM response"
        assert set(memo.sources) == {"http://web.example/a", "http://wiki.example/b"}

    def test_search_failure_does_not_crash_interview(self):
        graph = self.build(web_fail=True, wiki_fail=True)
        result = graph.invoke({"researcher": RESEARCHER, "topic": "espresso", "max_turns": 1})
        assert len(result["completed_research"]) == 1
        assert result["completed_research"][0].sources == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'agent'` (Task 2 tests still pass if run alone).

- [ ] **Step 3: Write `agent.py` (interview subgraph half)**

```python
"""LangGraph construction: research (interview) subgraph and parent report graph."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.graph import END, START, StateGraph

from models import CompletedResearch, InterviewState, SearchQuery

# --- Prompts -----------------------------------------------------------------

QUESTION_PROMPT = """You are a researcher with this persona:
{persona}

You are interviewing an expert to learn about: {topic}

Ask one insightful question that digs into your specific focus area. If the
conversation already contains answers, ask a follow-up that goes deeper or
fills a gap. Reply with only the question."""

SEARCH_PROMPT = """You will be given a conversation between a researcher and an
expert. Generate one well-formed web search query that would help answer the
researcher's latest question."""

ANSWER_PROMPT = """You are an expert being interviewed by a researcher. Answer
their latest question using ONLY the context below. Be specific, and mention
which document supports each claim where possible.

Context:
{context}"""

SUMMARY_PROMPT = """You are the researcher with this persona:
{persona}

Below is the transcript of your expert interview about: {topic}

Write a focused research memo (2-4 paragraphs) capturing the key findings from
your angle. Include the concrete facts from the interview."""


# --- Routing (module level for testability) ----------------------------------

def route_after_answer(state: InterviewState) -> str:
    """Loop back for another Q&A round until max_turns is reached."""
    if state.turn_count >= state.max_turns:
        return "save_research"
    return "ask_question"


# --- Interview subgraph -------------------------------------------------------

def build_interview_graph(llm, web_search, wiki_retriever):
    """One researcher's interview loop: question -> (web + wiki in parallel) -> answer."""

    def ask_question(state: InterviewState):
        system = SystemMessage(content=QUESTION_PROMPT.format(
            persona=state.researcher.persona, topic=state.topic))
        question = llm.invoke([system] + state.messages)
        return {"messages": [AIMessage(content=question.content, name="researcher")]}

    def _search_query(state: InterviewState) -> str:
        structured = llm.with_structured_output(SearchQuery)
        result = structured.invoke([SystemMessage(content=SEARCH_PROMPT)] + state.messages)
        return result.query

    def search_web(state: InterviewState):
        try:
            results = web_search.invoke({"query": _search_query(state)})["results"]
        except Exception as exc:  # a dead search API must not kill the parallel batch
            return {"context": [f"[web search unavailable: {exc}]"]}
        docs = [f'<Document href="{r["url"]}"/>\n{r["content"]}\n</Document>' for r in results]
        return {"context": docs, "sources": [r["url"] for r in results]}

    def search_wikipedia(state: InterviewState):
        try:
            results = wiki_retriever.invoke(_search_query(state))
        except Exception as exc:
            return {"context": [f"[wikipedia unavailable: {exc}]"]}
        docs = [
            f'<Document source="{d.metadata.get("source", "wikipedia")}"/>\n{d.page_content}\n</Document>'
            for d in results
        ]
        return {"context": docs,
                "sources": [d.metadata.get("source", "wikipedia") for d in results]}

    def generate_answer(state: InterviewState):
        system = SystemMessage(content=ANSWER_PROMPT.format(context="\n\n".join(state.context)))
        answer = llm.invoke([system] + state.messages)
        return {"messages": [AIMessage(content=answer.content, name="expert")],
                "turn_count": state.turn_count + 1}

    def save_research(state: InterviewState):
        transcript = get_buffer_string(state.messages)
        system = SystemMessage(content=SUMMARY_PROMPT.format(
            persona=state.researcher.persona, topic=state.topic))
        summary = llm.invoke([system, HumanMessage(content=transcript)])
        memo = CompletedResearch(researcher=state.researcher, summary=summary.content,
                                 sources=sorted(set(state.sources)))
        return {"completed_research": [memo]}

    builder = StateGraph(InterviewState)
    builder.add_node("ask_question", ask_question)
    builder.add_node("search_web", search_web)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("save_research", save_research)

    builder.add_edge(START, "ask_question")
    builder.add_edge("ask_question", "search_web")        # fan out: both searches
    builder.add_edge("ask_question", "search_wikipedia")  # run in the same superstep
    builder.add_edge("search_web", "generate_answer")
    builder.add_edge("search_wikipedia", "generate_answer")
    builder.add_conditional_edges("generate_answer", route_after_answer,
                                  ["ask_question", "save_research"])
    builder.add_edge("save_research", END)
    return builder.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add agent.py test_agent.py
git commit -m "feat: add interview subgraph with parallel web+wiki search"
```

---

### Task 4: Parent graph — personas, approval interrupt, fan-out, report (`agent.py`, part 2)

**Files:**
- Modify: `agent.py` (append parent-graph half)
- Modify: `test_agent.py` (append parent-graph section)

**Interfaces:**
- Consumes: `build_interview_graph`, `route_after_answer` (Task 3); all models (Task 2).
- Produces (used by Task 5):
  - `build_graph(model: str = "gpt-4o-mini", llm=None, web_search=None, wiki_retriever=None, checkpointer=None) -> CompiledStateGraph` — compiled WITH a checkpointer (default `InMemorySaver`), so `interrupt()` works.
  - `route_after_approval(state: ResearchState) -> str | list[Send]` (module level).
  - Interrupt payload shape: `{"researchers": [<Researcher.model_dump()>, ...], "message": str}`; resume value: `str` (empty = approve, non-empty = feedback).

- [ ] **Step 1: Append failing tests to `test_agent.py`**

Extend imports:

```python
from langgraph.types import Command, Send

from agent import build_graph, route_after_approval
```

Add the test class:

```python
class TestParentGraph:
    def build(self):
        return build_graph(llm=make_fake_llm(), web_search=make_fake_web_search(),
                           wiki_retriever=make_fake_wiki())

    @staticmethod
    def config():
        return {"recursion_limit": 100, "configurable": {"thread_id": "t1"}}

    def test_graph_has_expected_nodes(self):
        nodes = set(self.build().get_graph().nodes)
        assert {"create_researchers", "human_approval", "conduct_research",
                "write_intro", "write_body", "write_conclusion",
                "finalize_report"} <= nodes

    def test_route_after_approval_feedback_regenerates(self):
        state = ResearchState(topic="t", researchers=[RESEARCHER],
                              human_feedback="more diversity")
        assert route_after_approval(state) == "create_researchers"

    def test_route_after_approval_approved_fans_out(self):
        state = ResearchState(topic="t", max_turns=2,
                              researchers=[RESEARCHER, RESEARCHER])
        sends = route_after_approval(state)
        assert [s.node for s in sends] == ["conduct_research", "conduct_research"]
        assert all(isinstance(s, Send) for s in sends)
        assert sends[0].arg["topic"] == "t"
        assert sends[0].arg["max_turns"] == 2

    def test_run_pauses_at_approval_then_resumes_to_report(self):
        graph = self.build()
        config = self.config()
        result = graph.invoke({"topic": "espresso", "num_researchers": 2,
                               "max_turns": 1}, config)
        # paused at the human approval interrupt
        assert "__interrupt__" in result
        payload = result["__interrupt__"][0].value
        assert len(payload["researchers"]) == 2
        assert payload["researchers"][0]["name"] == "Ada Vale"
        # resume with empty feedback = approve
        result = graph.invoke(Command(resume=""), config)
        assert "__interrupt__" not in result
        assert len(result["completed_research"]) == 2  # one memo per researcher
        assert "fake LLM response" in result["final_report"]
        assert "## Sources" in result["final_report"]
        assert "http://web.example/a" in result["final_report"]

    def test_feedback_regenerates_and_pauses_again(self):
        graph = self.build()
        config = self.config()
        graph.invoke({"topic": "espresso", "num_researchers": 2, "max_turns": 1}, config)
        result = graph.invoke(Command(resume="swap the economist"), config)
        # regenerated team pauses at a fresh interrupt instead of finishing
        assert "__interrupt__" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: FAIL at import — `ImportError: cannot import name 'build_graph' from 'agent'`

- [ ] **Step 3: Append the parent graph to `agent.py`**

Extend the existing imports at the top of the file:

```python
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send, interrupt

from models import (CompletedResearch, InterviewState, Researcher,  # noqa: F401
                    ResearcherTeam, ResearchState, SearchQuery)
```

Append below the interview subgraph:

```python
# --- Parent-graph prompts ------------------------------------------------------

CREATE_RESEARCHERS_PROMPT = """You are assembling a team of {num} researcher
personas to investigate a topic.

Topic: {topic}

Previous human feedback on the proposed team (incorporate it if present):
{feedback}

Create exactly {num} researchers with distinct, complementary angles on the
topic. Give each a realistic name, a professional role, and a one-sentence
focus describing the specific angle they will investigate."""

INTRO_PROMPT = """Write a compelling introduction (a `# <title>` line then 1-2
paragraphs) for a research report on: {topic}

The report is based on these research memos:
{research}"""

BODY_PROMPT = """Write the main body of a research report on: {topic}

Synthesize these research memos from different researchers into coherent
markdown sections (## headers). Preserve concrete facts; do not invent
information.

{research}"""

CONCLUSION_PROMPT = """Write a concise conclusion (one paragraph, starting with
the line `## Conclusion`) for a research report on: {topic}

Research memos:
{research}"""


def format_research(completed: list[CompletedResearch]) -> str:
    return "\n\n".join(
        f"### Memo from {cr.researcher.name} ({cr.researcher.role})\n{cr.summary}"
        for cr in completed
    )


# --- Routing (module level for testability) ------------------------------------

def route_after_approval(state: ResearchState):
    """Non-empty feedback regenerates the team; approval fans out one
    interview subgraph per researcher via the Send API."""
    if state.human_feedback:
        return "create_researchers"
    return [
        Send("conduct_research", {
            "researcher": r,
            "topic": state.topic,
            "max_turns": state.max_turns,
        })
        for r in state.researchers
    ]


# --- Parent graph ----------------------------------------------------------------

def build_graph(model: str = "gpt-4o-mini", llm=None, web_search=None,
                wiki_retriever=None, checkpointer=None):
    """Full research workflow. Pass fakes for llm/web_search/wiki_retriever in tests."""
    llm = llm or ChatOpenAI(model=model, temperature=0)
    web_search = web_search or TavilySearch(max_results=3)
    wiki_retriever = wiki_retriever or WikipediaRetriever(top_k_results=2)
    interview_graph = build_interview_graph(llm, web_search, wiki_retriever)

    def create_researchers(state: ResearchState):
        prompt = CREATE_RESEARCHERS_PROMPT.format(
            num=state.num_researchers, topic=state.topic,
            feedback=state.human_feedback or "None")
        team = llm.with_structured_output(ResearcherTeam).invoke(
            [SystemMessage(content=prompt)])
        return {"researchers": team.researchers}

    def human_approval(state: ResearchState):
        feedback = interrupt({
            "researchers": [r.model_dump() for r in state.researchers],
            "message": "Press Enter to approve, or type feedback to regenerate.",
        })
        return {"human_feedback": feedback or None}

    def write_intro(state: ResearchState):
        msg = llm.invoke([SystemMessage(content=INTRO_PROMPT.format(
            topic=state.topic, research=format_research(state.completed_research)))])
        return {"intro": msg.content}

    def write_body(state: ResearchState):
        msg = llm.invoke([SystemMessage(content=BODY_PROMPT.format(
            topic=state.topic, research=format_research(state.completed_research)))])
        return {"body": msg.content}

    def write_conclusion(state: ResearchState):
        msg = llm.invoke([SystemMessage(content=CONCLUSION_PROMPT.format(
            topic=state.topic, research=format_research(state.completed_research)))])
        return {"conclusion": msg.content}

    def finalize_report(state: ResearchState):
        sources = sorted({s for cr in state.completed_research for s in cr.sources})
        sources_md = "\n".join(f"- {s}" for s in sources) or "- (none collected)"
        report = (f"{state.intro}\n\n{state.body}\n\n{state.conclusion}"
                  f"\n\n## Sources\n{sources_md}\n")
        return {"final_report": report}

    builder = StateGraph(ResearchState)
    builder.add_node("create_researchers", create_researchers)
    builder.add_node("human_approval", human_approval)
    builder.add_node("conduct_research", interview_graph)
    builder.add_node("write_intro", write_intro)
    builder.add_node("write_body", write_body)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    builder.add_edge(START, "create_researchers")
    builder.add_edge("create_researchers", "human_approval")
    builder.add_conditional_edges("human_approval", route_after_approval,
                                  ["create_researchers", "conduct_research"])
    builder.add_edge("conduct_research", "write_intro")       # report sections
    builder.add_edge("conduct_research", "write_body")        # are written
    builder.add_edge("conduct_research", "write_conclusion")  # in parallel
    builder.add_edge("write_intro", "finalize_report")
    builder.add_edge("write_body", "finalize_report")
    builder.add_edge("write_conclusion", "finalize_report")
    builder.add_edge("finalize_report", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add agent.py test_agent.py
git commit -m "feat: add parent graph with approval interrupt, Send fan-out, parallel report"
```

---

### Task 5: CLI (`research.py`) + smoke test

**Files:**
- Create: `research.py`
- Modify: `test_agent.py` (append CLI section)

**Interfaces:**
- Consumes: `build_graph` (Task 4); interrupt payload/resume convention from Task 4.
- Produces: `python research.py "<topic>" [--researchers N] [--max-turns N] [--model NAME]`; `require_env_keys(env)` and `parse_args(argv)` (module level, tested).

- [ ] **Step 1: Append failing tests to `test_agent.py`**

```python
from research import parse_args, require_env_keys


class TestCLI:
    def test_parse_args_defaults(self):
        args = parse_args(["quantum computing"])
        assert args.topic == "quantum computing"
        assert args.researchers == 3
        assert args.max_turns == 2
        assert args.model == "gpt-4o-mini"

    def test_parse_args_overrides(self):
        args = parse_args(["t", "--researchers", "5", "--max-turns", "1",
                           "--model", "gpt-4o"])
        assert (args.researchers, args.max_turns, args.model) == (5, 1, "gpt-4o")

    def test_require_env_keys_raises_naming_missing_keys(self):
        with pytest.raises(SystemExit) as exc:
            require_env_keys(env={"OPENAI_API_KEY": "sk-x"})
        assert "TAVILY_API_KEY" in str(exc.value)

    def test_require_env_keys_passes_when_all_present(self):
        require_env_keys(env={"OPENAI_API_KEY": "sk-x", "TAVILY_API_KEY": "tvly-x"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'research'`

- [ ] **Step 3: Write `research.py`**

```python
"""CLI entry point: python research.py "<topic>" [--researchers N] [--max-turns N] [--model NAME]"""
import argparse
import os
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from agent import build_graph

REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY")


def require_env_keys(env=os.environ):
    missing = [k for k in REQUIRED_KEYS if not env.get(k)]
    if missing:
        raise SystemExit(
            f"Missing required API keys: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parallel multi-researcher report agent (LangGraph)")
    parser.add_argument("topic", help="The topic to research")
    parser.add_argument("--researchers", type=int, default=3,
                        help="Number of researcher personas (default: 3)")
    parser.add_argument("--max-turns", type=int, default=2,
                        help="Q&A rounds per researcher (default: 2)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    return parser.parse_args(argv)


def print_team(payload):
    print("\nProposed researcher team:")
    for r in payload["researchers"]:
        print(f"  - {r['name']} ({r['role']}): {r['focus']}")


def main(argv=None):
    args = parse_args(argv)
    load_dotenv()
    require_env_keys()

    graph = build_graph(model=args.model)
    config = {"recursion_limit": 100,
              "configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"Researching: {args.topic}")
    print(f"  researchers={args.researchers}  max_turns={args.max_turns}  model={args.model}")
    result = graph.invoke({"topic": args.topic,
                           "num_researchers": args.researchers,
                           "max_turns": args.max_turns}, config)

    while "__interrupt__" in result:
        print_team(result["__interrupt__"][0].value)
        feedback = input("\nPress Enter to approve, or type feedback to regenerate: ").strip()
        if feedback:
            print("Regenerating the team with your feedback...")
        else:
            print("Approved. Researchers are working (this can take a minute)...")
        result = graph.invoke(Command(resume=feedback), config)

    print("\n" + "=" * 72 + "\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n langgraph-poc pytest test_agent.py -v`
Expected: 19 passed

- [ ] **Step 5: Manual smoke test (real APIs — requires filled-in `.env`)**

Run: `conda run -n langgraph-poc --no-capture-output python research.py "history of the espresso machine" --researchers 2 --max-turns 1`
Expected: prints a 2-persona team → press Enter → after ~1 minute prints a markdown report ending in a `## Sources` section with real URLs. If `.env` is not yet filled in, expect the clear `Missing required API keys` error instead, and defer this step to the user.

- [ ] **Step 6: Commit**

```bash
git add research.py test_agent.py
git commit -m "feat: add CLI with approval loop and report output"
```
