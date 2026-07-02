"""LangGraph construction: research (interview) subgraph and parent report graph."""
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from models import (CompletedResearch, InterviewState, Researcher,  # noqa: F401
                    ResearcherTeam, ResearchState, SearchQuery)

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
