import sys
import io
import traceback
import os

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI

from fastapi import FastAPI
from langserve import add_routes


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set in Render Environment Variables"
    )

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

    # These two fields are used by the web/LangServe UI
    task: Optional[str]
    command: Optional[str]


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    # Remove markdown code fences if Gemini returns them
    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""

    prompt = (
        f"You are a Senior QA Engineer. "
        f"Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. "
        f"Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# ============================================================
# 4. GRAPH NODES
# ============================================================

def task_input_node(state: CrewState):
    """
    Gets the coding task from the LangServe web request.
    This replaces the old input() function.
    """

    user_task = state.get("task")

    if not user_task:
        return {
            "next_step": "exit"
        }

    user_task = user_task.strip()

    if user_task.lower() == "exit":
        return {
            "next_step": "exit"
        }

    return {
        "messages": [
            HumanMessage(content=user_task)
        ],
        "next_step": "developer"
    }


# ------------------------------------------------------------
# DEVELOPER NODE
# ------------------------------------------------------------

def real_time_developer(state: CrewState):

    print("\n[Developer] Writing dynamic code using LLM...")

    # Get latest task
    task = state["messages"][-1].content

    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )

    if llm_flash is None:
        raise ValueError(
            "LLM is not initialized."
        )

    response = llm_flash.invoke(dev_prompt)

    # Safely parse Gemini content
    content = response.content

    if isinstance(content, list):

        if len(content) > 0:

            if isinstance(content[0], dict):
                code_str = content[0].get("text", "")
            else:
                code_str = str(content[0])

        else:
            code_str = ""

    else:
        code_str = str(content)

    # Remove markdown fences just in case
    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print(code_str)

    return {
        "code": code_str
    }


# ------------------------------------------------------------
# TESTER NODE
# ------------------------------------------------------------

def real_time_tester(state: CrewState):

    print("\n[Tester] Generating dynamic tests and executing code...")

    task = state["messages"][-1].content

    # Generate test cases
    test_cases = generate_test_cases.invoke(task)

    content = test_cases

    if isinstance(content, list):

        if len(content) > 0:

            if isinstance(content[0], dict):
                cases_str = content[0].get("text", "")
            else:
                cases_str = str(content[0])

        else:
            cases_str = ""

    else:
        cases_str = str(content)

    # Execute generated Python code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # Compile report
    report = (
        f"### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


# ------------------------------------------------------------
# MANAGER NODE
# ------------------------------------------------------------

def manager_decision_node(state: CrewState):

    print("\n" + "=" * 50)
    print("--- MANAGER DASHBOARD : TEST REPORT ---")

    print(
        state.get(
            "report",
            "No report available."
        )
    )

    print("=" * 50)

    # Get command from LangServe request
    user_input = state.get(
        "command",
        "store"
    )

    if user_input is None:
        user_input = "store"

    user_input = user_input.lower().strip()

    if user_input == "store":

        return {
            "next_step": "archiver"
        }

    else:

        return {
            "next_step": "task_input"
        }


# ------------------------------------------------------------
# ARCHIVER NODE
# ------------------------------------------------------------

def archiver_node(state: CrewState):

    print(
        "\n[Archiver] Task stored successfully. "
        "Closing workflow."
    )

    return {
        "next_step": "exit"
    }


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================

rt_workflow = StateGraph(CrewState)


# Add nodes
rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# ============================================================
# 6. GRAPH ROUTING
# ============================================================

# START -> task_input
rt_workflow.add_edge(
    START,
    "task_input"
)


# ------------------------------------------------------------
# Route from task input
# ------------------------------------------------------------

def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# ------------------------------------------------------------
# Developer -> Tester
# ------------------------------------------------------------

rt_workflow.add_edge(
    "developer",
    "tester"
)


# ------------------------------------------------------------
# Tester -> Manager
# ------------------------------------------------------------

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ------------------------------------------------------------
# Route from Manager
# ------------------------------------------------------------

def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":

        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ------------------------------------------------------------
# Archiver -> END
# ------------------------------------------------------------

rt_workflow.add_edge(
    "archiver",
    END
)


# ============================================================
# 7. COMPILE GRAPH
# ============================================================

rt_app = rt_workflow.compile()

print(
    "Interactive pipeline compiled and ready for live execution."
)


# ============================================================
# 8. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="LangGraph Developer Tester",
    description="AI Developer and Real-Time Testing Agent",
    version="1.0"
)


# Add LangGraph to LangServe
add_routes(
    app,
    rt_app,
    path="/agent"
)


# ============================================================
# 9. HOME PAGE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph LangServe API is running",
        "endpoint": "/agent",
        "playground": "/agent/playground/",
        "docs": "/docs/"
    }


# ============================================================
# 10. RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
