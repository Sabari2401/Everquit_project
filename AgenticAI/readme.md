# Agentic AI using LangGraph and Ollama

This project demonstrates an **Agentic AI workflow** using **LangGraph** with three specialized agents:
- Planner
- Executor
- Verifier

The agents collaborate to solve a user question in a structured, reliable way using the **LLaMA3 model via Ollama**.

---

## Architecture Overview

The system follows a **multi-step agent pipeline**:

1. **Planner Agent**
   - Generates step-by-step instructions to solve the problem
   - Does NOT solve the problem directly

2. **Executor Agent**
   - Solves the problem strictly based on the planner’s steps
   - Produces the final answer

3. **Verifier Agent**
   - Verifies whether the planner steps and executor answer are correct
   - Marks the result as `Success` or `Failed`

The flow is orchestrated using **LangGraph StateGraph**.

---

## Tech Stack

- Python
- LangGraph
- Ollama
- LLaMA3
- TypedDict (for shared state management)

---

## Project Structure

├── agentic_ai.py
├── requirements.txt
├── README.md


---

## State Definition

The shared state (`AgenticState`) contains:

- `question` – user input
- `plan` – step-by-step plan from planner
- `answer` – solution from executor
- `verified` – verifier feedback
- `status` – Success or Failed

---

## Installation

### 1. Install Python dependencies

pip install langgraph ollama

# install ollama

ollama pull llama3

# To run program:

python agentic_ai.py

# sample question:

Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?
