# Teaching Agent

An LLM-based teaching assistant designed to support adaptive learning from lecture materials.
The system integrates retrieval, long-term memory, and agent-based reasoning to provide slide-aware explanations and personalized learning feedback.

---

## Overview

Teaching Agent is an LLM-powered system designed to assist students in understanding course materials through document-grounded explanations and adaptive feedback.

The system combines retrieval-augmented generation (RAG), long-term student memory, and agent-based orchestration to create a closed-loop learning workflow. By tracking student interactions and responses, the agent can generate explanations aligned with lecture materials and maintain a persistent student learning state.

This project explores how LLM agents can support adaptive educational workflows through document understanding, retrieval, and feedback-driven reasoning.

---

## System Architecture

The system follows a modular agent-based architecture composed of several core components:

* **Teaching Agent** – orchestrates reasoning, retrieval, and learning feedback.
* **Retrieval Module** – retrieves relevant lecture content from indexed documents.
* **Student Memory** – stores interaction history and student learning state.
* **Assessment Module** – evaluates responses and updates the student model.

Together these components form a closed learning loop:

Student Question → Retrieval → Agent Reasoning → Explanation → Memory Update

---

## Key Features

**Document-Grounded Teaching**
Responses are generated with retrieval support to ensure alignment with lecture materials.

**Long-Term Student Memory**
Stores interaction history and learning signals to support personalized explanations.

**Agent-Based Workflow**
Implements an agent-driven pipeline that coordinates retrieval, reasoning, and feedback.

**Adaptive Feedback Generation**
Student responses and interaction patterns are used to adjust explanations and guidance.

---

## Tech Stack

* **LLM**: OpenAI GPT models
* **Agent Framework**: PydanticAI
* **Vector Database**: Chroma
* **Embedding Model**: OpenAI Embeddings
* **Backend**: Python

---

## Workflow

1. Course materials are parsed and converted into structured text.
2. Documents are chunked and indexed into a vector database.
3. When a student asks a question, the retrieval module searches for relevant content.
4. The teaching agent reasons over the retrieved context and generates an explanation.
5. The system stores interaction history in student memory for future reasoning.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourname/teaching-agent.git
cd teaching-agent

pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file and configure the required API keys:

```
OPENAI_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-large
VECTOR_DB=chroma
```

---

## Usage

Run the system:

```bash
python main.py
```

---

## Project Structure

```
teaching-agent
│
├── agents            # Agent definitions and orchestration
├── memory            # Student memory management
├── retrieval         # Document retrieval pipeline
├── prompts           # Prompt templates
├── services          # Core system services
└── main.py           # Entry point
```

---

## Future Work

* Improve student modeling using richer interaction signals
* Introduce multi-agent teaching workflows
* Add automated evaluation for question quality

---

## License

MIT License
