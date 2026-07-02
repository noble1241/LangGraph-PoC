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
