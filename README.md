# Teaching Agent (Self-Evolving Teaching Agent for Adaptive Practice)

A self-evolving teaching LLM agent for adaptive practice, built with **PydanticAI**.
The system combines **agentic RAG**, **dual long-term memory**, and **assessment-driven feedback loops** to deliver slide-aware, student-tailored teaching and continuously improve quiz quality metrics.

---

# Overview

This project builds a **self-evolving teaching agent** that supports adaptive learning workflows.

The system integrates **dual long-term memory**, **agentic retrieval**, and **automated evaluation signals** to provide slide-aware explanations and personalized practice.

Key capabilities include:

* **Slide-aware teaching** grounded in course materials
* **Student-tailored feedback** based on persistent learning state
* **Continuous self-improvement** driven by evaluation signals and memory updates

The agent forms a **closed learning loop** that continuously refines teaching strategies and quiz-bank quality based on student interaction signals.

---

# Core Ideas

## Dual Long-Term Memory

The agent maintains two persistent memory stores:

* **In-class student memory**
  Stores student interaction history, mistakes, and inferred mastery signals.

* **Cross-semester teaching memory**
  Stores reusable teaching knowledge and instructional artifacts across cohorts.

These memories are retrievable during reasoning and continuously updated through feedback signals.

---

## Agentic RAG over Memory + Student-State Summaries

The retrieval system operates over both **course materials** and **long-term memory**.

Retrieval pipelines combine:

* **BM25** for lexical keyword retrieval
* **HNSW-based ANN retrieval** for dense vector similarity search

To support efficient reasoning, the system maintains **compressed student-state summaries** that can be injected into prompts for in-context learning while controlling context length.

---

## Feedback-Driven Memory Update with Verification Guardrails

The system maintains a continuously updating memory pipeline:

1. Generate explanation or guidance grounded on retrieved evidence
2. Extract learning signals (errors, confidence, misconceptions)
3. Update student and teaching memory stores
4. Apply **fact-grounded verification guardrails** to prevent incorrect memory accumulation

This design reduces **memory drift** and stabilizes long-term learning signals.

---

# Quality Improvements (Automated Metrics)

Through iterative agent evaluation and quiz refinement, the system improved key quiz-bank metrics:

* **Discrimination Score:** 0.17 → 0.24
* **Error Concentration:** 0.36 → 0.52
* **Confidence:** 0.12 → 0.05

These metrics indicate improved question quality and stronger differentiation between student mastery levels.

---

# Tech Stack

* **Language:** Python
* **Agent Framework:** PydanticAI
* **Retrieval:** BM25 + dense retrieval (HNSW ANN indexing)
* **Vector Store:** Chroma
* **LLM / Embeddings:** OpenAI GPT models + OpenAI Embeddings

---

# High-Level Workflow

1. Ingest and index course materials and memory stores
2. Retrieve relevant context from course content and long-term memory
3. Inject compressed student-state summary into the reasoning context
4. Generate slide-aware explanations or practice questions
5. Update memory based on student interaction signals
6. Periodically evaluate and refine quiz-bank items using automated metrics

---

# Repository Structure

Adjust this section according to the actual repository layout.

```
.
├── agents/          # PydanticAI agent definitions and orchestration
├── retrieval/       # BM25 / dense retrieval pipelines
├── memory/          # long-term memory logic and summarization
├── assessment/      # scoring and feedback signals
├── evaluation/      # quiz-bank evaluation metrics
├── prompts/         # prompt templates
├── config/          # configuration loading
└── main.py          # system entry point
```

---

# Setup

Install the required dependencies.

```bash
cd pydantic-ai-tutorial
python3 -m pip install -r requirements.txt chromadb tqdm
```

Ensure the `.env` file in the project root contains your OpenAI API key:

```
OPENAI_API_KEY=your_api_key
```

---

# Running the Teaching Agent

The teaching agent is typically started using **two terminals**.

All commands should be executed inside:

```
pydantic-ai-tutorial/src
```

---

## Terminal 1 — Start the Teaching Agent Loop

Start the main agent loop:

```bash
cd pydantic-ai-tutorial/src
python3 watch_slide_adaptive.py
```

This process continuously monitors slide updates and triggers retrieval, reasoning, and question generation.

---

## Terminal 2 — Simulate Slide Changes

In a second terminal, simulate slide transitions by updating the slide file:

```bash
cd pydantic-ai-tutorial/src
echo "probability_foundations_and_bayes_rule" > current_slide.txt
```

You can also write natural language descriptions:

```bash
echo "Bayes rule prior and posterior" > current_slide.txt
```

Each update to `current_slide.txt` triggers a new retrieval and question generation cycle.

---

# Rebuilding the Quiz Index

If the system reports that questions cannot be retrieved or the index appears inconsistent, rebuild the quiz-bank index:

```bash
cd pydantic-ai-tutorial/src
python3 build_qbank_index_openai.py
```

After rebuilding the index, restart the Teaching Agent in **Terminal 1**.

---

# Future Work

Potential improvements include:

* richer student modeling signals
* more advanced multi-agent teaching workflows
* improved automatic question quality evaluation

---

# License

MIT License
