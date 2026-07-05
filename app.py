import streamlit as st

from agent import build_graph, merge_or_reset

st.set_page_config(page_title="Research Analyst Agent", page_icon="🔎", layout="wide")

st.title("🔎 Research Analyst Agent")
st.caption("Planner → parallel Researchers → Fact Checker (auto-retries on contradictions) → Writer")

# ---------------------------------------------------------------------------
# Sidebar: API keys
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("API Keys")
    groq_key = st.text_input("GROQ_API_KEY", type="password")
    tavily_key = st.text_input("TAVILY_API_KEY", type="password")
    st.divider()
    st.caption("Keys are only kept in this browser session and are never stored.")

topic = st.text_input("Research topic", placeholder="e.g. AI in Defense")
run_btn = st.button("Run research", type="primary", disabled=not (groq_key and tavily_key and topic))

if not (groq_key and tavily_key):
    st.info("Enter your GROQ and TAVILY API keys in the sidebar to get started.")

NODE_LABELS = {
    "orchestrator": "Starting up",
    "planner": "📋 Planning subtopics",
    "researcher": "🔬 Researching a subtopic",
    "combiner": "🧩 Combining research",
    "fact_checker": "✅ Fact-checking",
    "reset_for_retry": "♻️ Contradiction found — clearing for retry",
    "writer": "📝 Writing final report",
}


def merge_update_into_state(acc: dict, node_name: str, update: dict) -> dict:
    """Mirrors the graph's own reducers so we can track the full state locally
    while streaming, without needing a second graph.invoke() call."""
    for key, value in update.items():
        if key == "data":
            acc["data"] = merge_or_reset(acc.get("data", []), value)
        else:
            acc[key] = value
    return acc


if run_btn:
    graph = build_graph(groq_key, tavily_key)

    initial_state = {
        "topic": topic,
        "subtopics": [],
        "data": [],
        "combined_data": "",
        "next_agent": "planner",
        "final_report": "",
        "complete": False,
        "contradiction_points": "",
        "fact_status": True,
    }

    state_acc = dict(initial_state)
    retries = 0

    subtopics_box = st.empty()
    contradiction_box = st.empty()

    with st.status("Running the research pipeline...", expanded=True) as status:
        try:
            for step in graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in step.items():
                    state_acc = merge_update_into_state(state_acc, node_name, update)
                    st.write(NODE_LABELS.get(node_name, node_name))

                    if node_name == "planner" and state_acc.get("subtopics"):
                        subtopics_box.info("**Subtopics:** " + ", ".join(state_acc["subtopics"]))

                    if node_name == "fact_checker":
                        if state_acc.get("fact_status") is False:
                            retries += 1
                            contradiction_box.warning(
                                f"Retry #{retries} — contradictions found:\n\n{state_acc.get('contradiction_points')}"
                            )
                        else:
                            contradiction_box.success("Fact check passed — no contradictions found.")

            status.update(label="Done!", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            state_acc = None

    if state_acc and state_acc.get("final_report"):
        st.divider()
        st.subheader("Final Report")
        st.markdown(state_acc["final_report"])