import os
import io
import traceback
import contextlib
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langserve import add_routes

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    task: Optional[str]
    command: Optional[str]


# ============================================================
# RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:

    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()

    elif code.startswith("```"):
        code = code[3:].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    output_buffer = io.StringIO()

    try:

        with contextlib.redirect_stdout(output_buffer):
            exec(code, {"__name__": "__main__"})

        output = output_buffer.getvalue()

        if output.strip():
            return output.strip()

        return "Code executed successfully with no output."

    except Exception:
        return traceback.format_exc()


# ============================================================
# GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:

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

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


# ============================================================
# TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    task = state.get("task")

    if not task:

        return {
            "next_step": "exit",
            "messages": state.get("messages", [])
        }

    message = HumanMessage(content=task)

    messages = state.get("messages", [])

    return {
        "messages": messages + [message],
        "task": task,
        "next_step": "developer"
    }


# ============================================================
# DEVELOPER NODE
# ============================================================

def real_time_developer(state: CrewState):

    messages = state.get("messages", [])

    if not messages:

        return {
            "code": "",
            "next_step": "tester"
        }

    task_description = messages[-1].content

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

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))

        code_str = "\n".join(parts)

    else:
        code_str = str(content)

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
# TESTER NODE
# ============================================================

def real_time_tester(state: CrewState):

    task = state.get("task", "")
    code = state.get("code", "")

    test_cases = generate_test_cases.invoke(
        {
            "task_description": task
        }
    )

    execution_output = run_python_code.invoke(
        {
            "code": code
        }
    )

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
# MANAGER NODE
# ============================================================

def manager_decision_node(state: CrewState):

    command = state.get("command", "store")

    if command.lower().strip() == "another":

        return {
            "next_step": "task_input",
            "report": state.get("report", "")
        }

    return {
        "next_step": "archiver",
        "report": state.get("report", "")
    }


# ============================================================
# ARCHIVER
# ============================================================

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# ============================================================
# ROUTING
# ============================================================

def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return "exit"

    return "developer"


def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


# ============================================================
# BUILD LANGGRAPH
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


rt_workflow.add_edge(
    START,
    "task_input"
)

rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input,
    {
        "developer": "developer",
        "exit": END
    }
)

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)

rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision,
    {
        "archiver": "archiver",
        "task_input": "task_input"
    }
)

rt_workflow.add_edge(
    "archiver",
    END
)


rt_app = rt_workflow.compile()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Real-Time LangGraph Developer and Tester",
    version="1.0.0",
    description="LangGraph Developer and Tester deployed using LangServe"
)


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    rt_app,
    path="/agent"
)


# ============================================================
# WEB UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

    <title>LangGraph Developer & Tester</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1000px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-sizing: border-box;
            resize: vertical;
        }

        button {
            margin-top: 15px;
            padding: 12px 25px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background: #222;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #444;
        }

        .section {
            margin-top: 30px;
        }

        pre {
            background: #f1f1f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        .loading {
            display: none;
            margin-top: 20px;
        }

        .error {
            color: red;
            margin-top: 20px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Real-Time LangGraph Developer & Tester</h1>

    <div class="subtitle">
        Enter a programming task. Gemini Developer generates the code
        and the Tester generates and executes test cases.
    </div>


    <textarea
        id="task"
        placeholder="Example: Write a Python program to check whether a number is even or odd."
    ></textarea>


    <br>


    <button onclick="runAgent()">
        Run Developer & Tester
    </button>


    <div id="loading" class="loading">
        Running LangGraph... Please wait.
    </div>


    <div id="error" class="error"></div>


    <div class="section">

        <h2>Generated Python Code</h2>

        <pre id="code">Your generated code will appear here.</pre>

    </div>


    <div class="section">

        <h2>Test Report</h2>

        <pre id="report">Your test report will appear here.</pre>

    </div>

</div>


<script>

async function runAgent() {

    const task = document.getElementById("task").value;

    const loading = document.getElementById("loading");
    const error = document.getElementById("error");

    const code = document.getElementById("code");
    const report = document.getElementById("report");


    if (!task.trim()) {

        alert("Please enter a programming task.");

        return;
    }


    loading.style.display = "block";
    error.innerText = "";

    code.innerText = "Generating code...";
    report.innerText = "Generating test report...";


    try {

        const response = await fetch("/agent/invoke", {

            method: "POST",

            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },

            body: JSON.stringify({

                input: {

                    messages: [],

                    next_step: null,

                    code: null,

                    report: null,

                    task: task,

                    command: "store"

                }

            })

        });


        if (!response.ok) {

            throw new Error(
                "Server returned HTTP " + response.status
            );

        }


        const data = await response.json();


        code.innerText =
            data.output.code || "No code generated.";


        report.innerText =
            data.output.report || "No report generated.";


    }

    catch (err) {

        error.innerText =
            "Error: " + err.message;

        code.innerText = "";
        report.innerText = "";

    }

    finally {

        loading.style.display = "none";

    }

}

</script>


</body>

</html>
"""


# ============================================================
# RUN SERVER
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
