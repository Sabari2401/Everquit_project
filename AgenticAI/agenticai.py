from langgraph.graph import StateGraph,END,START
from typing import TypedDict,List,Optional
import ollama

class AgenticState(TypedDict):
    question: str
    answer : str
    plan :str
    verified :str
    status : str

def planner(state: AgenticState) -> AgenticState:
    prompt_=f""" you are excellent planner
    DO NOT solve the problem just give step by step execution to solve the problem
    
    question: {state["question"]}

    """
    response=ollama.chat(

        model='llama3',

        messages=[
            {
                "role":"user", "content":prompt_
             
            }
        ]
    )

    state["plan"]=response["message"]["content"]
    return state



def Executor(state:AgenticState) -> AgenticState:
    prompt=f"""you are excecuting agent. solve the user problem using the stepsmention in the plan.
    Give only the answer to the user

    plan={state["plan"]}
    question={state["question"]}
    
    """

    response=ollama.chat(

        model='llama3',
        messages=[
            {
                "role":"user", "content": prompt
            }
        ]
    )

    state["answer"]=response["message"]["content"]
    return state

def Verifier(state: AgenticState) -> AgenticState:
    prompt=f""" you are an Verifiying agent. Verifier the intermidiate steps given by the planner and compare the
    answer given by the llm and your answer

    question={state["question"]}
    plan={state["plan"]}
    llm_answer={state["answer"]}
    Output:
     - Confirm if correct or incorrect.
    - If incorrect, provide explanation. 
    """

    response=ollama.chat(
        model='llama3',
        messages=[
            {
                "role":"user", "content":prompt
            }
        ]
    )

    state["verified"]=response["message"]["content"]
    if "correct" in state["verified"].lower():
        state["status"]="Success"
    else:
        state["status"]="Failed"
    return state

    

# def llm(state: AgenticState) -> AgenticState:
#     # Call Ollama
#     response = ollama.chat(
#         model='llama3',
#         messages=[
#             {"role": "user", "content": state["question"]}
#         ]
#     )
    
#     # Save the answer into the state
#     state["answer"] = response["message"]["content"]
    
#     return state
question=input("enter the question:")

graph=StateGraph(AgenticState)

#graph.add_node("llm",llm)
graph.add_node("planner",planner)
graph.add_node("Executor",Executor)
graph.add_node("Verifier",Verifier)

graph.add_edge(START, "planner")
graph.add_edge("planner", "Executor")
graph.add_edge("Executor", "Verifier")
graph.add_edge("Verifier",END)
# Compile
app = graph.compile()

initial_state: AgenticState = {
    "question": question,
    "answer": ""
    
}

# Invoke the graph with the initial state
final_state = app.invoke(initial_state)

# Print the answer
print("\nLLM Answer:")
print("\nPlanning Steps:\n", final_state["plan"])
print("\nFinal Answer:\n", final_state["answer"])
print("\nFinal Answer:\n", final_state["verified"])
print("\nFinal Answer:\n", final_state["status"])