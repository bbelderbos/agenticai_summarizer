# Agentic AI Summarizer

A tiny, end-to-end demo of the pattern at the heart of most "AI agent" apps:

> **unstructured text → validated structured data → persist → display**

You give it a URL or a chunk of text; it returns a structured summary (TL;DR, key
points, tags, reading time, sentiment), optionally saving it to a local SQLite database.

## The spine

The interesting part is *not* the summarizing — it's how the LLM is forced to return
**structured, validated data** instead of free-form prose:

1. A Pydantic model (`SummaryResponse`) defines the exact shape we want.
2. Its JSON schema is handed to Claude as a **tool**, and the model is *required* to call
   that tool (`tool_choice`). The tool input is the structured summary.
3. We validate that input back into the Pydantic model. No regex, no "parse the prose."

Everything else is a thin shell around that idea:

| Module | Role |
|--------|------|
| `models.py` | `SummaryResponse` (LLM schema) + `Summary` (DB row) + `Sentiment` enum |
| `prompts.py` | The system prompt |
| `fetch.py` | Turn a URL into clean text (the one "tool" beyond the LLM) |
| `llms.py` | `Summarizer` protocol + `AnthropicSummarizer` (tool-calling, cost) |
| `repo.py` | Repository pattern: in-memory (tests) + SQLite (real) |
| `service.py` | Orchestrates fetch → summarize → persist |
| `cli.py` | A [cyclopts](https://cyclopts.readthedocs.io/) command-line interface |

## What this demo deliberately leaves out

To stay small and legible, it is **CLI-only** and ships **one LLM provider**. There's no web
API, no UI, no auth, no multi-provider switching. The `Summarizer` protocol is kept so you can
*see* where a second provider would plug in. The whole article is sent in one (truncated) prompt
— no chunking or retrieval. That's the natural next step, not part of this demo.

## Setup

```bash
uv sync
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Usage

```bash
# Summarize a URL
uv run summarizer summarize "https://example.com/some-article"

# Summarize pasted text, and save it to the database
uv run summarizer summarize "long text here..." --db

# List saved summaries
uv run summarizer list
```

## Tests

```bash
uv run pytest -q
```

Tests are deterministic — they mock the Anthropic client and the network, so no API key or
internet is needed.
