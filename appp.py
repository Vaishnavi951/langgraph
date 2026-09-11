import os
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. GEMINI API CONFIGURATION
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set in Render."
    )

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

    # These are used instead of input() for LangServe
    task: Optional[str]
    command: Optional[str]


# ============================================================
# 3. TOOL - RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Executes Python code and returns the output.
    """

    # Remove markdown code fences if Gemini returns them
    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()

    elif code.startswith("```"):
        code = code[3:].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    # Capture stdout
    old_stdout = io.StringIO()

    import contextlib

    try:
        with contextlib.redirect_stdout(old_stdout):

            # Execute the generated Python code
            exec(code, {"__name__": "__main__"})

        output = old_stdout.getvalue()

        if output.strip():
            return output.strip()

        return "Code executed successfully with no output."

    except Exception:
        return traceback.format_exc()


# ============================================================
# 4. TOOL - GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generates test scenarios for the given programming task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 meaningful test scenarios for the following
programming task.

Programming Task:
{task_description}

For every test scenario include:

1. Test Case Number
2. Input
3. Expected Output
4. Purpose

Keep the test cases clear and practical.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text_parts.append(
                    str(item.get("text", item))
                )
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# 5. TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    # LangServe cannot use terminal input().
    # Therefore, task is received from the API request.

    task = state.get("task")

    if not task:
        return {
            "next_step": "exit",
            "messages": state.get(
                "messages",
                []
            )
        }

    message = HumanMessage(content=task)

    messages = state.get("messages", [])

    return {
        "messages": messages + [message],
        "task": task,
        "next_step": "developer"
    }


# ============================================================
# 6. REAL-TIME DEVELOPER NODE
# ============================================================

def real_time_developer(state: CrewState):

    messages = state.get("messages", [])

    if not messages:
        return {
            "code": "",
            "next_step": "tester"
        }

    latest_message = messages[-1]

    task_description = latest_message.content

    prompt = f"""
You are an expert Python developer.

Write Python code to solve the following programming task.

TASK:
{task_description}

IMPORTANT RULES:

1. Return ONLY Python code.
2. Do not use Markdown.
3. Do not use ```python.
4. The program must be executable.
5. Include print statements so that the output can be tested.
6. Keep the solution simple and correct.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:

            if isinstance(item, dict):
                code_parts.append(
                    str(item.get("text", item))
                )

            else:
                code_parts.append(str(item))

        code_str = "\n".join(code_parts)

    else:
        code_str = str(content)

    # Remove Markdown fences if Gemini still adds them
    code_str = code_str.strip()

    if code_str.startswith("```python"):
        code_str = code_str[len("```python"):].strip()

    elif code_str.startswith("```"):
        code_str = code_str[3:].strip()

    if code_str.endswith("```"):
        code_str = code_str[:-3].strip()

    return {
        "code": code_str,
        "next_step": "tester"
    }


# ============================================================
# 7. REAL-TIME TESTER NODE
# ============================================================

def real_time_tester(state: CrewState):

    task = state.get("task", "")
    code = state.get("code", "")

    # Generate test cases
    test_cases = generate_test_cases.invoke(
        {
            "task_description": task
        }
    )

    # Execute generated code
    execution_output = run_python_code.invoke(
        {
            "code": code
        }
    )

    # Prepare report
    report = f"""
=============================
REAL-TIME SOFTWARE TEST REPORT
=============================

PROGRAMMING TASK:
{task}

=============================
GENERATED PYTHON CODE
=============================

{code}

=============================
TEST CASES
=============================

{test_cases}

=============================
EXECUTION RESULT
=============================

{execution_output}

=============================
END OF REPORT
=============================
"""

    return {
        "report": report,
        "next_step": "manager"
    }


# ============================================================
# 8. MANAGER DECISION NODE
# ============================================================

def manager_decision_node(state: CrewState):

    report = state.get("report", "")

    # In LangServe we cannot use input().
    # The command comes from the API request.

    command = state.get("command", "store")

    command = command.lower().strip()

    if command == "another":
        return {
            "next_step": "task_input",
            "report": report
        }

    return {
        "next_step": "archiver",
        "report": report
    }


# ============================================================
# 9. ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# ============================================================
# 10. CONDITIONAL ROUTING
# ============================================================

def route_from_input(state: CrewState):

    next_step = state.get("next_step")

    if next_step == "exit":
        return "exit"

    return "developer"


def route_from_decision(state: CrewState):

    next_step = state.get("next_step")

    if next_step == "archiver":
        return "archiver"

    return "task_input"


# ============================================================
# 11. BUILD LANGGRAPH WORKFLOW
# ============================================================

rt_workflow = StateGraph(CrewState)

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


# START → TASK INPUT

rt_workflow.add_edge(
    START,
    "task_input"
)


# TASK INPUT → DEVELOPER / END

rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input,
    {
        "developer": "developer",
        "exit": END
    }
)


# DEVELOPER → TESTER

rt_workflow.add_edge(
    "developer",
    "tester"
)


# TESTER → MANAGER

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# MANAGER → ARCHIVER / TASK INPUT

rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision,
    {
        "archiver": "archiver",
        "task_input": "task_input"
    }
)


# ARCHIVER → END

rt_workflow.add_edge(
    "archiver",
    END
)


# COMPILE GRAPH

rt_app = rt_workflow.compile()


# ============================================================
# 12. FASTAPI APPLICATION FOR LANGSERVE
# ============================================================

app = FastAPI(
    title="Real-Time LangGraph Developer and Tester",
    version="1.0.0",
    description="LangGraph workflow deployed using LangServe"
)


# ============================================================
# 13. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    rt_app,
    path="/agent"
)


# ============================================================
# 14. ROOT ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph LangServe API is running",
        "endpoint": "/agent",
        "playground": "/agent/playground/"
    }


# ============================================================
# 15. LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
