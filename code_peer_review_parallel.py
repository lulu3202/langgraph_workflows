import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatGroq(model="qwen-2.5-32b")

# Graph state
class State(TypedDict):
    code_snippet: str  # input
    readability_feedback: str  # intermediate
    security_feedback: str  # intermediate
    best_practices_feedback: str  # intermediate
    feedback_aggregator: str  # output


# Nodes
@traceable
def get_readability_feedback(state: State):
    """First LLM call to check code readability"""
    msg = llm.invoke(
        f"Provide readability feedback for the following code:\n\n {state['code_snippet']}"
    )
    return {"readability_feedback": msg.content}


@traceable
def get_security_feedback(state: State):
    """Second LLM call to check for security vulnerabilities in code"""
    msg = llm.invoke(
        f"Check for potential security vulnerabilities in the following code and provide feedback:\n\n {state['code_snippet']}"
    )
    return {"security_feedback": msg.content}


@traceable
def get_best_practices_feedback(state: State):
    """Third LLM call to check for adherence to coding best practices"""
    msg = llm.invoke(
        f"Evaluate the adherence to coding best practices in the following code and provide feedback:\n\n {state['code_snippet']}"
    )
    return {"best_practices_feedback": msg.content}


@traceable
def aggregate_feedback(state: State):
    """Combine all the feedback from the three LLM calls into a single output"""
    combined = f"Here's the overall feedback for the code:\n\n"
    combined += f"READABILITY FEEDBACK:\n{state['readability_feedback']}\n\n"
    combined += f"SECURITY FEEDBACK:\n{state['security_feedback']}\n\n"
    combined += f"BEST PRACTICES FEEDBACK:\n{state['best_practices_feedback']}"
    return {"feedback_aggregator": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes - Corrected node names
parallel_builder.add_node("get_readability_feedback", get_readability_feedback)
parallel_builder.add_node("get_security_feedback", get_security_feedback)
parallel_builder.add_node("get_best_practices_feedback", get_best_practices_feedback)
parallel_builder.add_node("aggregate_feedback", aggregate_feedback)

# Add edges to connect nodes
parallel_builder.add_edge(START, "get_readability_feedback")
parallel_builder.add_edge(START, "get_security_feedback")
parallel_builder.add_edge(START, "get_best_practices_feedback")
parallel_builder.add_edge("get_readability_feedback", "aggregate_feedback")
parallel_builder.add_edge("get_security_feedback", "aggregate_feedback")
parallel_builder.add_edge("get_best_practices_feedback", "aggregate_feedback")
parallel_builder.add_edge("aggregate_feedback", END)

parallel_workflow = parallel_builder.compile()

# Show workflow - Try and except to handle the timeout
try:
    display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Error generating Mermaid diagram: {e}")

# Invoke
# Here is an example of a program, you can change it for any python code.
full_program = """
import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("LANGCHAIN_API_KEY"))
"""
state = parallel_workflow.invoke({"code_snippet": full_program})
print(state["feedback_aggregator"])

