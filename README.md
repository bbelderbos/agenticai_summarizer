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

### Example output

```text
$ uv run summarizer summarize https://belderbos.dev/blog/htmx-hx-swap-oob-django/ --db
                                                         Summary
┌──────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Source       │ https://belderbos.dev/blog/htmx-hx-swap-oob-django/                                                     │
│ TL;DR        │ This article explains how to use htmx's out-of-band swaps (hx-swap-oob) to update multiple unrelated    │
│              │ page elements from a single HTTP request, avoiding the need to duplicate server-side logic in          │
│              │ JavaScript.                                                                                            │
│ Key points   │ • Out-of-band swaps allow htmx to update elements outside the main target by using the hx-swap-oob      │
│              │ attribute on elements in the response.                                                                 │
│              │ • Use hx-swap-oob="innerHTML:#selector" to update an element's inner content without replacing the      │
│              │ element itself, preserving attached event listeners.                                                   │
│              │ • Centralizing data fetching and rendering logic in the view prevents code duplication and maintains a  │
│              │ single source of truth for business rules.                                                             │
│ Tags         │ htmx, web development, django, python, oob-swaps, frontend architecture                                 │
│ Reading time │ 5 min                                                                                                  │
│ Sentiment    │ positive                                                                                               │
│ Cost         │ $0.004491                                                                                              │
│ Saved        │ yes                                                                                                    │
└──────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Tests

```bash
uv run pytest -q
```

Tests are deterministic — they mock the Anthropic client and the network, so no API key or
internet is needed.
