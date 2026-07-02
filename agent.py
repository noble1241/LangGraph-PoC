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
from prompts import (ANSWER_PROMPT, BODY_PROMPT, CONCLUSION_PROMPT,
                     CREATE_RESEARCHERS_PROMPT, INTRO_PROMPT, QUESTION_PROMPT,
                     SEARCH_PROMPT, SUMMARY_PROMPT)

MAX_REPORT_SOURCES = 10

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
        shown, remaining = sources[:MAX_REPORT_SOURCES], len(sources) - MAX_REPORT_SOURCES
        sources_md = "\n".join(f"- {s}" for s in shown) or "- (none collected)"
        if remaining > 0:
            sources_md += f"\n- ...and {remaining} more"
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
