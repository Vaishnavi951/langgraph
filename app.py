import os
import io
import sys
import traceback
import uvicorn
from typing import List, Optional
from typing_extensions import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    task: Optional[str]
    command: Optional[str]

def run_python_code(code: str) -> str:
    clean_code = code.strip()

    if clean_code.startswith("```python"):
        clean_code = clean_code[len("```python"):].strip()
    elif clean_code.startswith("```"):
        clean_code = clean_code[3:].strip()

    if clean_code.endswith("```"):
        clean_code = clean_code[:-3].strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        execution_scope = {"__name__": "__main__"}
        exec(clean_code, execution_scope, execution_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = "Execution Error:\n" + traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    result = result.strip()
    return result if result else "Success (no terminal output)"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in Render Environment Variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

def generate_test_cases(task_description: str) -> str:
    prompt = f"""
You are a software quality assurance engineer.

The user has requested the following Python coding task:

{task_description}

Generate a simple set of test cases that can be used
to evaluate the generated Python program.

Include:
1. Normal case
2. Boundary case
3. Special case if applicable
4. Expected behavior

Keep the test cases simple and easy to understand.

Do not generate Python code.
Return only the test-case description.
"""
    response = llm.invoke(prompt)
    return response.content

def task_input_node(state: CrewState):
    task = state.get("task")

    if not task:
        return {"next_step": "end", "report": "No coding task was provided."}

    return {"next_step": "developer"}

def route_from_input(state: CrewState):
    if state.get("next_step") == "developer":
        return "developer"
    return END

def real_time_developer(state: CrewState):
    task = state.get("task", "")

    prompt = f"""
You are an expert Python developer.

Coding task:

{task}

Write a complete Python program that solves the task.

IMPORTANT RULES:

1. Return ONLY the Python code.
2. Do NOT use Markdown.
3. Do NOT use ```python.
4. Include print() statements so the output can be observed.
5. Do NOT use input().
6. Since this program will be executed automatically on a server,
   use a simple example/test value directly inside the program
   whenever user input would normally be required.
7. Make sure the program can execute without interactive input.
8. Write clean and simple Python code.
9. Make sure the program is syntactically correct.
"""

    response = llm.invoke(prompt)
    generated_code = response.content.strip()

    if generated_code.startswith("```python"):
        generated_code = generated_code[len("```python"):].strip()
    elif generated_code.startswith("```"):
        generated_code = generated_code[3:].strip()

    if generated_code.endswith("```"):
        generated_code = generated_code[:-3].strip()

    return {
        "code": generated_code,
        "messages": [HumanMessage(content=generated_code)],
        "next_step": "tester"
    }

def real_time_tester(state: CrewState):
    task = state.get("task", "")
    code = state.get("code", "")

    test_cases = generate_test_cases(task)
    execution_result = run_python_code(code)

    report = f"""
==============================
QA TEST REPORT
==============================

CODING TASK:
{task}

==============================
GENERATED CODE
==============================

{code}

==============================
GENERATED TEST CASES
==============================

{test_cases}

==============================
EXECUTION RESULT
==============================

{execution_result}

==============================
END OF TEST REPORT
==============================
"""

    return {"report": report, "next_step": "manager"}

def manager_decision_node(state: CrewState):
    command = state.get("command", "store").lower().strip()

    if command == "another":
        return {"next_step": "another"}

    return {"next_step": "store"}

def route_from_decision(state: CrewState):
    decision = state.get("next_step")

    if decision == "store":
        return "archiver"

    if decision == "another":
        return END

    return END

def archiver_node(state: CrewState):
    task = state.get("task", "")
    code = state.get("code", "")
    report = state.get("report", "")

    archive_message = f"""
==============================
ARCHIVED DEVELOPER-TESTER RUN
==============================

TASK:
{task}

CODE:
{code}

REPORT:
{report}

==============================
ARCHIVE COMPLETE
==============================
"""

    return {"report": archive_message, "next_step": "end"}

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("manager_decision", manager_decision_node)
rt_workflow.add_node("archiver", archiver_node)

rt_workflow.add_edge(START, "task_input")
rt_workflow.add_conditional_edges("task_input", route_from_input)
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", "manager_decision")
rt_workflow.add_conditional_edges("manager_decision", route_from_decision)
rt_workflow.add_edge("archiver", END)

rt_app = rt_workflow.compile()

class AgentInput(BaseModel):
    task: str = Field(description="Coding task for the Developer and Tester")
    command: str = Field(default="store", description="Manager command: store or another")

def format_for_agent(x):
    if isinstance(x, AgentInput):
        task = x.task
        command = x.command
    else:
        task = x.get("task", "")
        command = x.get("command", "store")

    return {
        "messages": [],
        "next_step": None,
        "code": None,
        "report": None,
        "task": task,
        "command": command
    }

class AgentOutput(BaseModel):
    code: str = Field(description="Generated Python code")

def extract_code(state):
    return {"code": state.get("code", "No code was generated.")}

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | rt_app
    | RunnableLambda(extract_code)
).with_types(input_type=AgentInput, output_type=AgentOutput)

app = FastAPI(title="LangGraph Developer Tester")

add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

@app.get("/")
def home():
    return {
        "message": "LangGraph Developer Tester API is running",
        "endpoint": "/agent",
        "playground": "/agent/playground/",
        "docs": "/docs/"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
