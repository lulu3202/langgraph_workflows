import streamlit as st
import os
from typing import Literal, List, Dict, TypedDict, Annotated
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import Send
import operator
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- Helper Functions ---

def markdown_converter(text):
    return st.markdown(text)


# --- Blog Evaluator Workflow ---

class BlogState(TypedDict):
    topic: str
    blog: str
    evaluation: str
    feedback: str
    accepted: bool


def generate_blog(state: BlogState, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates short blogs."),
        ("human", "Generate a short blog about: {topic}")
    ])
    chain = prompt | llm
    result = chain.invoke({"topic": state["topic"]}).content
    return {"blog": result}


def evaluate_blog(state: BlogState, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict blog evaluator."),
        ("human",
         "Evaluate this blog:\n{blog}\nIs it concise, engaging, structured with subtitles and a conclusion? Respond with 'yes' or 'no'."),
        ("human", "If the answer is no. provide specific feedback on the needed improvements")
    ])
    chain = prompt | llm
    result = chain.invoke({"blog": state["blog"]}).content

    lines = result.split('\n')
    evaluation_text = lines[0].strip().lower()
    if 'no' in evaluation_text:
        return {"evaluation": "Needs Revision", "feedback": "\n".join(lines[1:]), "accepted": False}
    else:
        return {"evaluation": "Accepted", "feedback": "", "accepted": True}


def provide_feedback(state: BlogState):
    return {"feedback": state["feedback"]}


def conditional_check(state):
    if not state["accepted"]:
        return "revise"
    else:
        return "end"


def build_blog_graph(llm):
    def generate_blog_llm(state):
        return generate_blog(state, llm)

    def evaluate_blog_llm(state):
        return evaluate_blog(state, llm)

    graph = StateGraph(BlogState)
    graph.add_node("generate_blog", generate_blog_llm)
    graph.add_node("evaluate_blog", evaluate_blog_llm)
    graph.add_node("provide_feedback", provide_feedback)
    graph.set_entry_point("generate_blog")
    graph.add_conditional_edges(
        "evaluate_blog",
        conditional_check,
        {
            "revise": "generate_blog",
            "end": END
        }
    )
    graph.add_edge("generate_blog", "evaluate_blog")
    graph.add_edge("provide_feedback", "generate_blog")

    return graph


# --- Parallelized Code Review Workflow ---

class CodeReviewState(TypedDict):
    code_snippet: str
    readability_feedback: str
    security_feedback: str
    best_practices_feedback: str
    feedback_aggregator: str


@traceable
def get_readability_feedback(state: CodeReviewState, llm):
    """First LLM call to check code readability"""
    st.session_state.progress_text = "Analyzing Readability..."
    msg = llm.invoke([
        HumanMessage(content=f"Provide readability feedback for the following code:\n\n {state['code_snippet']}")
    ])
    return {"readability_feedback": msg.content}


@traceable
def get_security_feedback(state: CodeReviewState, llm):
    """Second LLM call to check for security vulnerabilities in code"""
    st.session_state.progress_text = "Analyzing Security..."
    msg = llm.invoke([
        HumanMessage(
            content=f"Check for potential security vulnerabilities in the following code and provide feedback:\n\n {state['code_snippet']}")
    ])
    return {"security_feedback": msg.content}


@traceable
def get_best_practices_feedback(state: CodeReviewState, llm):
    """Third LLM call to check for adherence to coding best practices"""
    st.session_state.progress_text = "Analyzing Best Practices..."
    msg = llm.invoke([
        HumanMessage(
            content=f"Evaluate the adherence to coding best practices in the following code and provide feedback:\n\n {state['code_snippet']}")
    ])
    return {"best_practices_feedback": msg.content}


@traceable
def aggregate_feedback(state: CodeReviewState):
    """Combine all the feedback from the three LLM calls into a single output"""
    st.session_state.progress_text = "Aggregating Feedback..."
    combined = f"Here's the overall feedback for the code:\n\n"
    combined += f"READABILITY FEEDBACK:\n{state['readability_feedback']}\n\n"
    combined += f"SECURITY FEEDBACK:\n{state['security_feedback']}\n\n"
    combined += f"BEST PRACTICES FEEDBACK:\n{state['best_practices_feedback']}"
    return {"feedback_aggregator": combined}


def build_code_review_graph(llm):
    def get_readability_feedback_llm(state):
        return get_readability_feedback(state, llm)

    def get_security_feedback_llm(state):
        return get_security_feedback(state, llm)

    def get_best_practices_feedback_llm(state):
        return get_best_practices_feedback(state, llm)

    parallel_builder = StateGraph(CodeReviewState)

    # Add nodes
    parallel_builder.add_node("get_readability_feedback", get_readability_feedback_llm)
    parallel_builder.add_node("get_security_feedback", get_security_feedback_llm)
    parallel_builder.add_node("get_best_practices_feedback", get_best_practices_feedback_llm)
    parallel_builder.add_node("aggregate_feedback", aggregate_feedback)

    # Add edges
    parallel_builder.add_edge(START, "get_readability_feedback")
    parallel_builder.add_edge(START, "get_security_feedback")
    parallel_builder.add_edge(START, "get_best_practices_feedback")
    parallel_builder.add_edge("get_readability_feedback", "aggregate_feedback")
    parallel_builder.add_edge("get_security_feedback", "aggregate_feedback")
    parallel_builder.add_edge("get_best_practices_feedback", "aggregate_feedback")
    parallel_builder.add_edge("aggregate_feedback", END)

    return parallel_builder.compile()


# --- Learning Path Generator Workflow ---

class Topic(BaseModel):
    name: str = Field(description="Name of the learning topic.")
    description: str = Field(description="Brief overview of the topic.")


class Topics(BaseModel):
    topics: List[Topic] = Field(description="List of topics to learn.")


class State(TypedDict):
    user_skills: str
    user_goals: str
    topics: List[Topic]
    completed_topics: Annotated[List[str], operator.add]
    learning_roadmap: str


class WorkerState(TypedDict):
    topic: Topic
    completed_topics: List[str]


@traceable
def orchestrator(state: State, planner):
    study_plan = planner.invoke([
        SystemMessage(
            content="Create a detailed study plan based on user skills and goals."
        ),
        HumanMessage(
            content=f"User skills: {state['user_skills']}\nUser goals: {state['user_goals']}"
        ),
    ])
    return {"topics": study_plan.topics}


@traceable
def llm_call(state: WorkerState, llm):
    topic_summary = llm.invoke([
        SystemMessage(
            content="Generate a content summary for the provided topic."
        ),
        HumanMessage(
            content=f"Topic: {state['topic'].name}\nDescription: {state['topic'].description}"
        ),
    ])

    return {"completed_topics": [topic_summary.content]}


@traceable
def synthesizer(state: State):
    topic_summaries = state["completed_topics"]
    learning_roadmap = "\n\n---\n\n".join(topic_summaries)
    return {"learning_roadmap": learning_roadmap}


def assign_workers(state: State):
    return [Send("llm_call", {"topic": t}) for t in state["topics"]]


def build_learning_path_graph(llm, planner):
    def orchestrator_planner(state):
        return orchestrator(state, planner)

    def llm_call_llm(state):
        return llm_call(state, llm)

    learning_path_builder = StateGraph(State)

    learning_path_builder.add_node("orchestrator", orchestrator_planner)
    learning_path_builder.add_node("llm_call", llm_call_llm)
    learning_path_builder.add_node("synthesizer", synthesizer)

    learning_path_builder.set_entry_point("orchestrator")
    learning_path_builder.add_conditional_edges("orchestrator", assign_workers, {"llm_call": "llm_call"})
    learning_path_builder.add_edge("llm_call", "synthesizer")
    learning_path_builder.add_edge("synthesizer", END)

    return learning_path_builder


# --- Streamlit App ---

st.set_page_config(page_title="LLM-Powered Workflows", layout="wide")

# Custom CSS for colors
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            background-color: #FF7F50; /* Coral */
        }
        [data-testid="stAppViewContainer"] {
          background-color: #FF1493; /* Deep Pink */
        }
        
        /* Adjusting main content text color */
        .block-container {
          color: #9400D3; /* Dark Violet */
        }
         /* for all text */
        body {
          color: #9400D3 !important; /* Dark Violet */
        }

    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Try out LLM-Powered Workflows")
st.markdown("""
    <p style='color:#9400D3; font-size: 20px;'>
        <b>1. Learning Path Generator</b> - Orchestrator-Synthesizer Workflow<br>
        <b>2. Peer Code Review</b> - Parallelized Workflow<br>
        <b>3. Blog Generation</b> - Evaluator-Optimizer Workflow
    </p>
    <p style='color:#9400D3;'><b>Enter your GROQ API key on the left to get started!</b></p>
    """, unsafe_allow_html=True)

# Initialize session state
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "mixtral-8x7b-32768"
if "progress_text" not in st.session_state:
    st.session_state.progress_text = ""
if "api_key_submitted" not in st.session_state:
    st.session_state.api_key_submitted = False
# Sidebar for API key, model selection, and workflow selection
with st.sidebar:
    st.header("Configuration")
    groq_api_key_input = st.text_input("Enter your Groq API Key:", type="password", key="api_key_input")
    api_key_submitted = st.button("Submit API Key")
    
    available_models = ["mixtral-8x7b-32768", "deepseek-r1-distill-qwen-32b", "qwen-2.5-32b", "llama-3.1-8b-instant"]
    
    llm = None
    planner = None

    if api_key_submitted:
        st.session_state.api_key_submitted = True

    if st.session_state.api_key_submitted:
        if groq_api_key_input:
            os.environ["GROQ_API_KEY"] = groq_api_key_input
        elif os.environ.get("GROQ_API_KEY"):
            groq_api_key_input = os.environ.get("GROQ_API_KEY")

        if groq_api_key_input or os.environ.get("GROQ_API_KEY"):
            try:
                llm = ChatGroq(groq_api_key=groq_api_key_input, model_name=st.session_state.model_choice)
                planner = llm.with_structured_output(Topics)
                st.success(f"API key loaded successfully!")
            
                st.session_state.model_choice = st.selectbox(
                "Choose a Model",
                available_models,
                key="model_select_box",
                index=available_models.index(st.session_state.model_choice) if st.session_state.model_choice in available_models else 0
                )

                llm = ChatGroq(groq_api_key=groq_api_key_input, model_name=st.session_state.model_choice)
                planner = llm.with_structured_output(Topics)

                st.success(f"model '{st.session_state.model_choice}' loaded successfully!")

            except Exception as e:
                 st.error(f"Error initializing LLM: {e}")
                 llm = None
                 planner = None
        else:
            st.warning("Please enter your Groq API key to continue.")
        
    if llm is not None:
        # Emojis for workflow choices
        workflow_emojis = {
            "Learning Path Generator": "📚 Learning Path",  # Books
            "Parallelized Code Review": "👨‍💻 Code Review",  # Man technologist
            "Blog Evaluator": "📝 Blog Evaluator",  # Writing hand
        }

        # Correct order for selectbox:
        workflow_order = ["Learning Path Generator", "Parallelized Code Review", "Blog Evaluator"]

        workflow_choice = st.selectbox(
            "Choose a Workflow",
            workflow_order,
            format_func=lambda x: f"{workflow_emojis[x]}",
            key="workflow_choice"
        )

# Main content area
if llm and planner:
    # Emojis for workflow choices
    workflow_emojis = {
            "Learning Path Generator": "📚",  # Books
            "Parallelized Code Review": "👨‍💻",  # Man technologist
            "Blog Evaluator": "📝",  # Writing hand
        }
    
    if st.session_state.get("workflow_choice") == "Learning Path Generator":
        st.header(f"{workflow_emojis['Learning Path Generator']} Learning Path Generator")
        user_skills = st.text_area("Enter your current skills:")
        user_goals = st.text_area("Enter your learning goals:")
        if st.button("Generate Learning Path"):
            if user_skills and user_goals:
                learning_graph = build_learning_path_graph(llm, planner)
                learning_app = learning_graph.compile()
                result = learning_app.invoke({"user_skills": user_skills, "user_goals": user_goals})
                st.subheader("Learning Roadmap:")
                markdown_converter(result["learning_roadmap"])
            else:
                st.error("Please enter both your skills and goals")
                
    elif st.session_state.get("workflow_choice") == "Parallelized Code Review":
        st.header(f"{workflow_emojis['Parallelized Code Review']} Parallelized Code Review")
        code_snippet = st.text_area("Enter code snippet:", height=300)
        review_button = st.button("Review Code")

        if review_button:
            if code_snippet:
                workflow = build_code_review_graph(llm)
                progress_bar = st.progress(0)
                progress_bar.progress(25, text="Starting...")
                result = workflow.invoke({"code_snippet": code_snippet})
                progress_bar.progress(100, text="Done!")
                st.subheader("Code Review Feedback:")
                st.markdown(result["feedback_aggregator"])
                progress_bar.empty()
                st.session_state.progress_text = ""
            else:
                st.error("Please enter a code snippet to review.")
        else:
            st.write(st.session_state.progress_text)
            
    elif st.session_state.get("workflow_choice") == "Blog Evaluator":
        st.header(f"{workflow_emojis['Blog Evaluator']} Blog Evaluator")
        blog_topic = st.text_input("Enter blog topic:")
        if st.button("Generate and Evaluate"):
            if blog_topic:
                blog_graph = build_blog_graph(llm)
                blog_app = blog_graph.compile()
                result = blog_app.invoke({"topic": blog_topic})
                st.subheader("Blog:")
                markdown_converter(result["blog"])
                #only display blog content. No Evaluation or feedback.
            else:
                st.error("Please enter a blog topic")
