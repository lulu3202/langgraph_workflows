Data Flow Breakdown in learning_path_orchestrator
We follow a structured data pipeline where each step modifies and passes data to the next stage.

1️⃣ Define Custom Data Structures
- Topic (BaseModel) → Represents a single topic with name and description.
- Topics (BaseModel) → A wrapper around multiple Topic objects (essentially a list of topics).
- State (TypedDict) → Holds global state, including user input, generated topics, and completed topics.
- WorkerState (TypedDict) → Holds individual topic assignments for processing.

2️⃣ Step-by-Step Data Flow

Step 1: Orchestrator Generates Topics
Input: user_skills and user_goals
Process: Calls planner.invoke(), which uses an LLM (Groq API) to generate topics.
Output: A structured Topics object (a list of Topic objects).
Storage: The topics list is saved inside State.
Returns: {"topics": study_plan.topics}
📌 Key Detail:
The Orchestrator only generates topics and doesn’t process them. It assigns each topic to workers.

Step 2: Assign Workers to Each Topic
Function: assign_workers(state: State)
Process: Iterates over state["topics"] and assigns each topic to a worker (i.e., llm_call).
Returns: A list of dispatch instructions, sending each topic to the llm_call function.
Key Mechanism:
Uses Send("llm_call", {"topic": t}), which maps each topic to WorkerState.
📌 Key Detail:
This step distributes work in parallel across multiple workers, each handling a single topic.

Step 3: LLM Call Generates Topic Summaries
Function: llm_call(state: WorkerState)
Input: A single topic object (from WorkerState).
Process:
Calls the LLM (llm.invoke) with the topic's name and description.
Generates a summary + resources in markdown format.
Output:
{"completed_topics": [topic_summary.content]}
Storage: The summaries are stored inside completed_topics in State.
📌 Key Detail:
Each worker only receives one topic at a time. The WorkerState helps isolate one topic per call instead of processing everything at once.

Step 4: Synthesizer Combines Summaries into a Learning Roadmap
Function: synthesizer(state: State)
Input: completed_topics list (all processed topics).
Process: Joins all summaries together into a structured format.
Output: {"learning_roadmap": learning_roadmap}
Final Storage: The roadmap is stored inside State.
📌 Key Detail:
This step aggregates all topic summaries into a final, structured learning plan.

3️⃣ Where Does the Data Go?
Step	Function	Input	Output	Where the Data Goes
1	orchestrator(state)	User skills & goals	topics list	Stored in State["topics"]
2	assign_workers(state)	Topics list	Send("llm_call", {"topic": t})	Sends each topic to llm_call
3	llm_call(state)	A single topic	{"completed_topics": [summary]}	Appends to State["completed_topics"]
4	synthesizer(state)	completed_topics list	learning_roadmap	Stores final roadmap in State["learning_roadmap"]

📝 Key Takeaways
- Orchestrator generates the topics based on user_skills and user_goals.
- Workers process each topic separately (using llm_call).
- WorkerState ensures only one topic is processed per worker to avoid mixing topics.
- The synthesizer combines all results into a final structured roadmap.
- Data flows in a structured manner through State and WorkerState, ensuring modular and parallel execution.