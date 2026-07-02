"""Prompt templates for the interview subgraph and the parent report graph."""

# --- Interview subgraph ---------------------------------------------------------

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


# --- Parent graph ------------------------------------------------------------------

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
