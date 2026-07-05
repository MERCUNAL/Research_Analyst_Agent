# Research Analyst Agent

A multi-agent research pipeline built with **LangGraph** + **Groq** + **Tavily**.

Given a topic, it plans subtopics, researches each one in parallel, fact-checks
the combined findings (auto-retrying research if contradictions are found),
and writes a final formatted report.

## How it works

```
orchestrator → planner --(fan out, one per subtopic)--> researcher → combiner → fact_checker
                                                                                     │
                                                        fact_status = True ──┐   fact_status = False
                                                                             │        │
                                                                          writer   reset_for_retry --(fan out again)--> researcher
                                                                             │
                                                                            END
```

- **planner** — splits the topic into subtopics (max 2), returned as a JSON list.
- **researcher** — runs once *per subtopic*, in parallel, using a Tavily-backed
  web search tool. On a retry round it also receives the fact checker's
  contradiction notes so it can correct itself.
- **combiner** — fan-in step that waits for every researcher branch and joins
  their output into one block of text.
- **fact_checker** — checks the combined research for contradictions.
  - No contradictions → routes to **writer**.
  - Contradictions found → routes to **reset_for_retry**, which clears the
    research data and sends the researchers out again.
- **writer** — turns the verified, combined research into a formatted report.

## Files

| File | Purpose |
|---|---|
| `agent.ipynb` | Original notebook version of the pipeline — good for exploring/debugging step by step. |
| `agent_graph.py` | The pipeline logic, refactored into a `build_graph(groq_key, tavily_key)` function so it can be imported. |
| `streamlit_app.py` | A Streamlit UI on top of `agent_graph.py` — enter API keys, type a topic, watch the pipeline run live, read the final report. |
| `requirements.txt` | Python dependencies. |

## Setup

1. Get API keys:
   - **Groq**: https://console.groq.com
   - **Tavily**: https://tavily.com

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the notebook

Open `agent.ipynb` and add your keys via a `.env` file in the same folder:
```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```
Then run all cells. The final report and pipeline state will print at the
bottom.

## Running the Streamlit app

```bash
streamlit run streamlit_app.py
```

No `.env` file needed here — paste your `GROQ_API_KEY` and `TAVILY_API_KEY`
into the sidebar (kept in-session only, never written to disk), enter a
topic, and click **Run research**. You'll see each pipeline step logged live,
including any retry rounds with the contradictions that triggered them, and
the final report rendered at the bottom.

## Notes / known constraints

- The planner is currently capped at **2 subtopics** — adjust the prompt in
  `planner()` (in `agent_graph.py` or the notebook) if you want more.
- `add_node(..., input_schema=...)` is used for the `researcher` node's
  smaller input schema. If you're on an older `langgraph` version, this
  argument may instead be named `input=`.
- Groq's free tier has fairly low tokens-per-minute limits — if you hit a
  `413 rate_limit_exceeded` error, try a shorter topic, fewer subtopics, or
  a smaller model.
- The fact checker's contradiction check is only as good as the model's
  judgment — it's a helpful pass, not a guarantee of factual accuracy.
