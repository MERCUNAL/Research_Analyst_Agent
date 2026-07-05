"""
Research analyst agent graph: planner -> parallel researchers -> combiner ->
fact_checker (retries research on contradictions) -> writer.

This is the same pipeline built in the notebook, refactored into a
`build_graph()` function so it can be imported by the Streamlit app.
"""

import json
from typing import Literal, Annotated

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from tavily import TavilyClient


def merge_or_reset(left: list, right: list) -> list:
    """Reducer for `data`: normally appends, but a ['__RESET__'] update clears it."""
    if right == ["__RESET__"]:
        return []
    return left + right


class State(BaseModel):
    topic: str
    subtopics: list = []
    data: Annotated[list, merge_or_reset] = []
    combined_data: str = ""
    next_agent: Literal["END", "planner", "researcher", "combiner", "fact_checker", "writer"] = "planner"
    final_report: str = ""
    complete: bool = False
    contradiction_points: str = ""
    fact_status: bool = True


class ResearcherInput(BaseModel):
    """Slim schema for the fanned-out researcher branches (one per subtopic)."""
    subtopic: str
    contradiction_points: str = ""


def build_graph(groq_api_key: str, tavily_api_key: str, model: str = "openai/gpt-oss-20b"):
    """Builds and compiles the LangGraph pipeline using the given API keys."""

    llm = ChatGroq(model=model, groq_api_key=groq_api_key)
    tavily_client = TavilyClient(api_key=tavily_api_key)

    @tool
    def web_search(query: str) -> str:
        """Search the web for up-to-date information."""
        result = tavily_client.search(query=query, max_results=5)
        return str(result)

    research_agent = create_agent(model=llm, tools=[web_search])

    # ---- AGENT: orchestrator ----
    def orchestrator(state):
        return {"topic": state.topic, "next_agent": "planner"}

    # ---- AGENT 1: research planner ----
    def planner(state):
        task = state.topic
        plan_prompt = f"""
        You are a research planner. Your task is to divide the given topic into a list of subtopics that can be researched independently.
        The topic is: {task}.

        Return STRICTLY a JSON array of subtopic strings and nothing else - no markdown, no commentary.
        Example: ["Subtopic 1", "Subtopic 2"]
        - At max give only two subtopics
        """
        plan_response = llm.invoke([HumanMessage(content=plan_prompt)])
        content = plan_response.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[len("json"):]
            content = content.strip()

        try:
            subtopics = json.loads(content)
        except json.JSONDecodeError:
            subtopics = [task]

        return {"subtopics": subtopics, "next_agent": "researcher"}

    # ---- AGENT 2: web researcher (fanned out, one call per subtopic) ----
    def researcher(payload):
        subtopic = payload["subtopic"]
        contradiction_points = payload.get("contradiction_points", "")

        correction_note = ""
        if contradiction_points:
            correction_note = f"""
        A previous fact-check pass found the following contradictions in earlier research on this topic:
        {contradiction_points}

        Please specifically re-check and correct these points using fresh web search results.
        """

        prompt = f"""
        You are a web researcher.

        Research: {subtopic}
        {correction_note}

        Include:
        1. Key facts and background
        2. Current trends or developments
        3. Important statistics or data points
        4. Notable examples or case studies

        Use the available web search tool whenever needed.
        Only upto 100 words.

        Return ONLY valid JSON.
        """

        result = research_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"data": [result["messages"][-1].content]}

    def fan_out_to_researchers(state):
        """Turns the subtopic list into one Send per subtopic (parallel fan-out)."""
        return [
            Send("researcher", {"subtopic": s, "contradiction_points": state.contradiction_points})
            for s in state.subtopics
        ]

    # ---- combiner: fan-in, waits for every researcher branch ----
    def combiner(state):
        combined = "\n\n---\n\n".join(state.data)
        return {"combined_data": combined, "next_agent": "fact_checker"}

    # ---- AGENT 3: fact checker ----
    def fact_checker(state):
        task = state.combined_data
        fact_prompt = f"""
        You are a fact checker. Your task is to fact check every data point and spot contradictions in : {task}.

        - If no contradictions are found, respond with EXACTLY: NO_CONTRADICTIONS
        - If contradictions are found, describe them clearly in plain text (which points conflict, and why).
        """
        fact_response = llm.invoke([HumanMessage(content=fact_prompt)])
        content = fact_response.content.strip()

        if "NO_CONTRADICTIONS" in content:
            return {"fact_status": True, "contradiction_points": "", "next_agent": "writer"}

        return {"fact_status": False, "contradiction_points": content, "next_agent": "researcher"}

    def reset_for_retry(state):
        """Clears `data` before re-running the researchers on a retry round."""
        return {"data": ["__RESET__"]}

    def router(state):
        return "writer" if state.fact_status else "reset_for_retry"

    # ---- AGENT 4: writer ----
    def writer(state):
        data = state.combined_data
        topic = state.topic
        subtopics = state.subtopics

        writer_prompt = f"""
        You are a writer. Your task is to create a research document on the data provided: {data} based on the topic : {topic}.
        - The data is divided based on the subtopics: {subtopics}.
        - Output a properly formatted string along with the topic on top with proper sizes of the headings and subheadings.
        Be concise but thorough.
        """
        writer_response = llm.invoke([HumanMessage(content=writer_prompt)])
        return {"final_report": writer_response.content, "next_agent": "END"}

    workflow = StateGraph(State)
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher, input_schema=ResearcherInput)
    workflow.add_node("combiner", combiner)
    workflow.add_node("fact_checker", fact_checker)
    workflow.add_node("reset_for_retry", reset_for_retry)
    workflow.add_node("writer", writer)

    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "planner")
    workflow.add_conditional_edges("planner", fan_out_to_researchers, ["researcher"])
    workflow.add_edge("researcher", "combiner")
    workflow.add_edge("combiner", "fact_checker")
    workflow.add_conditional_edges(
        "fact_checker",
        router,
        {"writer": "writer", "reset_for_retry": "reset_for_retry"},
    )
    workflow.add_conditional_edges("reset_for_retry", fan_out_to_researchers, ["researcher"])
    workflow.add_edge("writer", END)

    return workflow.compile()