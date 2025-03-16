import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display, Markdown
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import Send
import operator

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize LLM model
llm = ChatGroq(model="qwen-2.5-32b")

# Define Custom Data structures
class Topic(BaseModel):
    """Represents a learning topic with a name and description."""
    name: str = Field(description="Name of the learning topic.")
    description: str = Field(description="Brief overview of the topic.")

class Topics(BaseModel):
    """Wrapper for a list of learning topics."""
    topics: List[Topic] = Field(description="List of topics to learn.")

# Augment the LLM with structured schema
planner = llm.with_structured_output(Topics)

# Define the state that carries data throughout the workflow
class State(TypedDict):
    user_skills: str  
    user_goals: str  
    topics: List[Topic]  
    completed_topics: Annotated[List[str], operator.add]  # Merging completed topics
    learning_roadmap: str  

# Worker state for topic processing
class WorkerState(TypedDict):
    topic: Topic
    completed_topics: List[str]

# Define Node Functions
@traceable
def orchestrator(state: State):
    """Creates a study plan based on user skills and goals."""
    
    # LLM generates a structured study plan
    study_plan = planner.invoke([
        SystemMessage(
            content="Create a detailed study plan based on user skills and goals."
        ),
        HumanMessage(
            content=f"User skills: {state['user_skills']}\nUser goals: {state['user_goals']}"
        ),
    ])

    print("Study Plan:", study_plan)

    return {"topics": study_plan.topics}  # Returns generated topics


@traceable
def llm_call(state: WorkerState):
    """Generates a content summary for a specific topic."""
    
    # LLM processes the topic and generates a summary
    topic_summary = llm.invoke([
        SystemMessage(
            content="Generate a content summary for the provided topic."
        ),
        HumanMessage(
            content=f"Topic: {state['topic'].name}\nDescription: {state['topic'].description}"
        ),
    ])

    return {"completed_topics": [topic_summary.content]}  # Returns generated summary


@traceable
def synthesizer(state: State):
    """Compiles topic summaries into a structured learning roadmap."""
    
    topic_summaries = state["completed_topics"]
    learning_roadmap = "\n\n---\n\n".join(topic_summaries)  # Formatting output

    return {"learning_roadmap": learning_roadmap}  # Returns final roadmap


# Define Conditional Edge Function 

def assign_workers(state: State):
    """Assigns a worker (llm_call) to each topic in the plan."""
    
    return [Send("llm_call", {"topic": t}) for t in state["topics"]]  # Creates worker tasks


# Build Workflow

learning_path_builder = StateGraph(State)

# Add nodes
learning_path_builder.add_node("orchestrator", orchestrator)
learning_path_builder.add_node("llm_call", llm_call)
learning_path_builder.add_node("synthesizer", synthesizer)

# Define execution order using edges
learning_path_builder.add_edge(START, "orchestrator")  # Start with orchestrator
learning_path_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])  # Assign workers
learning_path_builder.add_edge("llm_call", "synthesizer")  # Process topics
learning_path_builder.add_edge("synthesizer", END)  # End workflow

# Compile workflow
learning_path_workflow = learning_path_builder.compile()

# ----------------------------
# 5️⃣ Run the Workflow
# ----------------------------

user_skills = "Python programming, basic machine learning concepts"
user_goals = "Learn advanced AI, master prompt engineering, and build AI applications"

state = learning_path_workflow.invoke(
    {"user_skills": user_skills, "user_goals": user_goals}
)

# Display the final learning roadmap
Markdown(state["learning_roadmap"])
