"""Tests for the research agent. All tests run offline: LLM and search tools are mocked."""
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from models import (
    CompletedResearch,
    InterviewState,
    Researcher,
    ResearcherTeam,
    ResearchState,
    SearchQuery,
)
from agent import build_interview_graph, route_after_answer

RESEARCHER = Researcher(name="Ada Vale", role="Materials scientist", focus="battery chemistry")


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
