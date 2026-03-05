# Teaching Agent (Self-Evolving Teaching Agent for Adaptive Practice)

A self-evolving teaching LLM agent for adaptive practice, built with **PydanticAI**.  
The system combines **agentic RAG**, **dual long-term memory**, and **assessment-driven feedback loops** to deliver slide-aware, student-tailored teaching and continuously improve quiz quality metrics.

---

## Overview

This project builds a **self-evolving teaching agent** that supports adaptive learning workflows:

- **Slide-aware teaching**: grounds explanations and guidance in course materials.
- **Student-tailored feedback**: conditions responses on a persistent student state.
- **Continuous self-improvement**: uses automated assessment signals to refine memory and practice content over time.

The agent is designed to function as a closed-loop system that learns from interactions and improves its instructional behavior and quiz-bank quality.

---

## Core Ideas

### Dual Long-Term Memory

The agent maintains two long-term memory stores:

- **In-class student memory**: persistent student-specific signals (e.g., recent questions, mistakes, inferred mastery).
- **Cross-semester teaching memory**: persistent teaching knowledge and reusable instructional artifacts across cohorts.

These memories are updated through a feedback loop and are retrievable for downstream reasoning.

---

### Agentic RAG over Memory + Student-State Summaries

The agent retrieves context from both course materials and memory:

- **Retrieval workflows**: combines lexical and vector-based retrieval:
  - **BM25** for keyword / sparse matching
  - **HNSW-based ANN** retrieval for dense vector search
- **Student-state compression**: maintains a compressed student-state summary that can be injected into prompts for in-context learning, reducing context length while preserving learning-relevant signals.

---

### Feedback-Driven Memory Update with Verification Guardrails

The system implements a continuously updating memory pipeline:

1. Generate response / guidance grounded on retrieved evidence
2. Extract learning signals (e.g., errors, confidence, misconceptions)
3. Update memory stores based on feedback
4. Apply **fact-grounded verification guardrails** to reduce accumulation of incorrect memory entries

This design aims to prevent “memory drift” by ensuring updates are supported by evidence.

---

## Quality Improvements (Automated Metrics)

Through iterative agent evaluation and quiz refinement, the system improved automated quiz-bank quality metrics:

- **Discrimination Score**: **0.17 → 0.24**
- **Error Concentration**: **0.36 → 0.52**
- **Confidence**: **0.12 → 0.05**

> Metric definitions and the evaluation protocol should be documented in `evaluation/` (recommended) to ensure reproducibility.

---

## Tech Stack

- **Language**: Python
- **Agent Framework**: PydanticAI
- **Retrieval**: BM25 + dense retrieval with HNSW-based ANN indexing
- **Vector Store / Memory Store**: Chroma
- **LLM / Embeddings**: OpenAI GPT models + OpenAI Embeddings

---

## High-Level Workflow

1. Ingest and index course materials / long-term memory
2. Retrieve relevant context (materials + memory)
3. Inject compressed student-state summary into the agent context
4. Generate slide-aware explanation / practice / feedback
5. Update memory with verification guardrails
6. Periodically evaluate and refine quiz-bank items using automated metrics

---

## Repository Structure (Suggested)

> Adjust to match your repo structure.

```text
.
├── agents/            # PydanticAI agent definitions and orchestration
├── retrieval/         # BM25 / dense retrieval pipelines and indexing
├── memory/            # long-term memory stores + update logic + summarization
├── assessment/        # scoring / evaluation hooks and feedback signals
├── evaluation/        # metric computation and evaluation scripts
├── prompts/           # prompt templates
├── config/            # configuration and env loading
└── main.py            # entry point
