import os
import io
import sys
import traceback
import uvicorn

from typing import List, Optional
from typing_extensions import TypedDict

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    task: Optional[str]
    command: Optional[str]


# ============================================================
# 2. PYTHON CODE EXECUTION
# ============================================================

def run_python_code(code: str) -> str:

    clean_code = code.strip()

    # Remove Markdown code fences if Gemini returns them
    if clean_code.startswith("```python"):
        clean_code = clean_code[len("```python"):].strip()

    if clean_code.startswith("```"):
        clean_code = clean_code[3:].strip()

    if clean_code.endswith("```"):
        clean_code = clean_code[:-3].strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        # Same dictionary for globals and locals.
        # This allows recursive functions such as factorial()
        # to work correctly.
        execution_scope = {
            "__name__": "__main__"
        }

        exec(
            clean_code,
            execution_scope,
            execution_scope
        )

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():
        return result.strip()

    return "Success (no terminal output)"


# ============================================================
# 3. GEMINI MODEL
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured in Render Environment Variables."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 4. TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    user_task = state.get("task")

    if not user_task:
        return {
            "next_step": "end"
        }

    return {
        "messages": [
            HumanMessage(content=user_task)
        ],
        "next_step": "developer"
    }


# ============================================================
# 5. DEVELOPER NODE
# ============================================================

def real_time_developer(state: CrewState):

    messages = state.get("messages", [])

    if not messages:
        return {
            "next_step": "tester"
        }

    task = messages[-1].content

    developer_prompt = f"""
You are the Developer in a Developer-Tester
software development workflow.

Coding task:
{task}

Write a complete Python program that solves the task.

Rules:

1. Return ONLY Python code.
2. Do not return explanations.
3. Do not use Markdown code fences.
4. The code must be syntactically correct.
5. The code must be executable.
6. Include print statements for the result.
7. Avoid unnecessary external libraries.
8. Do NOT use input() because the program will be
   automatically executed by a server-side tester.
9. Use a simple example value inside the program when
   input is required.
"""

    response = llm.invoke(developer_prompt)

    generated_code = response.content

    if not isinstance(generated_code, str):
        generated_code = str(generated_code)

    return {
        "code": generated_code,
        "next_step": "tester"
    }


# ============================================================
# 6. TESTER NODE
# ============================================================

def real_time_tester(state: CrewState):

    code = state.get("code")

    if not code:

        return {
            "report": "No code was generated.",
            "next_step": "manager_decision"
        }

    tester_prompt = f"""
You are the Tester in a Developer-Tester workflow.

Analyze this Python program:

{code}

Generate suitable test cases.

For every test case give:

1. Test case number
2. Input
3. Expected output
4. Purpose

Also identify whether the generated program
appears correct.

Return the testing information in plain text.
"""

    response = llm.invoke(tester_prompt)

    test_cases = response.content

    if not isinstance(test_cases, str):
        test_cases = str(test_cases)

    # Execute generated code
    execution_result = run_python_code(code)

    report = f"""
GENERATED TEST CASES
====================

{test_cases}


PROGRAM EXECUTION RESULT
========================

{execution_result}
"""

    return {
        "report": report,
        "next_step": "manager_decision"
    }


# ============================================================
# 7. MANAGER DECISION NODE
# ============================================================

def manager_decision_node(state: CrewState):

    command = state.get("command")

    if command is None:
        command = "store"

    command = command.lower().strip()

    if command == "another":

        return {
            "next_step": "task_input"
        }

    return {
        "next_step": "archiver"
    }


# ============================================================
# 8. ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):

    code = state.get("code", "")
    report = state.get("report", "")

    archive_message = f"""
==================================================
                 TASK ARCHIVED
==================================================

GENERATED CODE
--------------------------------------------------

{code}

TEST REPORT
--------------------------------------------------

{report}

==================================================
"""

    print(archive_message)

    return {
        "next_step": "end"
    }


# ============================================================
# 9. ROUTING FUNCTIONS
# ============================================================

def route_from_input(state: CrewState):

    next_step = state.get("next_step")

    if next_step == "developer":
        return "developer"

    return END


def route_from_decision(state: CrewState):

    next_step = state.get("next_step")

    if next_step == "task_input":
        return "task_input"

    if next_step == "archiver":
        return "archiver"

    return END


# ============================================================
# 10. BUILD LANGGRAPH
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


# START → TASK INPUT

rt_workflow.add_edge(
    START,
    "task_input"
)


# TASK INPUT → DEVELOPER

rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
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


# MANAGER → TASK INPUT / ARCHIVER

rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ARCHIVER → END

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile

rt_app = rt_workflow.compile()

print(
    "Interactive Developer-Tester LangGraph "
    "compiled and ready for execution."
)


# ============================================================
# 11. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(TypedDict):

    messages: List[BaseMessage]

    next_step: Optional[str]

    code: Optional[str]

    report: Optional[str]

    task: Optional[str]

    command: Optional[str]


# ============================================================
# 12. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):

        return {
            "messages": x.get("messages", []),
            "next_step": x.get("next_step"),
            "code": x.get("code"),
            "report": x.get("report"),
            "task": x.get("task"),
            "command": x.get("command")
        }

    return x


# ============================================================
# 13. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | rt_app
)


# ============================================================
# 14. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph Developer Tester"
)


# ============================================================
# 15. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 16. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
