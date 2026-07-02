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
from langgraph.types import Command, Send

from agent import build_graph, build_interview_graph, route_after_answer, route_after_approval

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

    def test_empty_team_ends_without_report(self):
        llm = make_fake_llm()
        llm.with_structured_output.side_effect = None
        structured = MagicMock()
        structured.invoke.return_value = ResearcherTeam(researchers=[])
        llm.with_structured_output.return_value = structured
        graph = build_graph(llm=llm, web_search=make_fake_web_search(),
                            wiki_retriever=make_fake_wiki())
        config = self.config()
        graph.invoke({"topic": "espresso", "num_researchers": 2, "max_turns": 1}, config)
        result = graph.invoke(Command(resume=""), config)
        assert result.get("final_report") is None

    def test_topic_survives_the_state_reducer(self):
        # Regression test: the ResearchState.topic/max_turns reducers must keep
        # the incoming value, not the field's pre-write zero-value. A reducer
        # of `lambda a, b: a` silently discards every write (topic ends up ""),
        # which a MagicMock LLM can't catch since it ignores prompt content.
        graph = self.build()
        config = self.config()
        graph.invoke({"topic": "espresso", "num_researchers": 1, "max_turns": 3}, config)
        state = graph.get_state(config).values
        assert state["topic"] == "espresso"
        assert state["max_turns"] == 3

    def test_finalize_report_caps_sources_list(self):
        from agent import MAX_REPORT_SOURCES

        llm = make_fake_llm()
        many_sources = [f"http://example.com/{i}" for i in range(MAX_REPORT_SOURCES + 5)]
        memo = CompletedResearch(researcher=RESEARCHER, summary="s", sources=many_sources)
        graph = self.build()
        state = ResearchState(topic="t", researchers=[RESEARCHER],
                              completed_research=[memo], intro="i", body="b",
                              conclusion="c")
        finalize_report = graph.get_graph().nodes["finalize_report"].data
        result = finalize_report.invoke(state)
        report = result["final_report"]
        assert report.count("http://example.com/") == MAX_REPORT_SOURCES
        assert "...and 5 more" in report


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

    def test_parse_args_rejects_zero_researchers(self):
        with pytest.raises(SystemExit):
            parse_args(["t", "--researchers", "0"])
