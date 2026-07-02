# LangGraph-PoC — Parallel Multi-Researcher Report Agent

A LangGraph research agent: give it a topic and a number of researchers.
It generates researcher personas (you approve or give feedback), then each
researcher interviews the web (Tavily) and Wikipedia in parallel, and the
findings are synthesized into a report written to your terminal.

## Setup

Option A — from the environment file:

```bash
conda env create -f environment.yml
```

Option B — bare conda env + pip requirements:

```bash
conda create -n langgraph-poc python=3.11 -y
conda activate langgraph-poc
pip install -r requirements.txt
```

Then in either case:

```bash
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
