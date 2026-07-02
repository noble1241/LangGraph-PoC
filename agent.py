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
