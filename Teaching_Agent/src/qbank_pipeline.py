"""
qbank_pipeline.py — Fully automated QBank lifecycle orchestrator.

Replaces qbank_manager.py. Uses qbank_agent as the primary engine for
question generation and quality evaluation, plus real-world student data
analysis for continuous improvement.

Usage:
    # Full pipeline: PDF → MCQ → Sim-Eval → Deploy (ChromaDB)
    python src/qbank_pipeline.py generate

    # Quality check: analyze real student data → auto-evolve flagged questions
    python src/qbank_pipeline.py maintain

    # Just rebuild ChromaDB index from current QBank JSON
    python src/qbank_pipeline.py reindex
"""

import argparse
import copy
import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import chromadb

from hybrid_retrieval import build_qbank_search_text

# ---------------------
# Ensure qbank_agent is importable
# ---------------------
os.environ.setdefault(
    "QBANK_PROJECT_ROOT",
    str(Path(__file__).parent.parent.parent / "QBank-agent")
)

from qbank_agent import config as qa_config
from qbank_agent import pdf_parser, syllabus_gen, mcq_gen, evaluation, utils

# ---------------------
# Local config (runtime paths within pydantic-ai-tutorial/src)
# ---------------------
SRC_DIR = Path(__file__).parent.resolve()
QBANK_JSON = SRC_DIR / "Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_8.json"
EVENTS_FILE = SRC_DIR / "data" / "student_events.jsonl"
CHROMA_DIR = SRC_DIR / "chroma_qbank_openai"
COLLECTION_NAME = "weijia_qbank"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64

# Quality thresholds (migrated from qbank_manager.py)
DISCRIMINATION_THRESHOLD = 0.15
DIFFICULTY_TOO_EASY = 0.90
DIFFICULTY_TOO_HARD = 0.10
MIN_ATTEMPTS_FOR_EVAL = 3


# ---------------------
# Data Models
# ---------------------
class QuestionQuality(BaseModel):
    topic_id: str
    question_id: str
    total_attempts: int = 0
    correct: int = 0
    item_difficulty: float = 0.0
    discrimination: float = 0.0
    flag: str = ""


# =====================
# Part A: Full Generation Pipeline
# =====================

def run_full_pipeline():
    """
    End-to-end: PDF → Parse → Syllabus → MCQ Gen → Shuffle → Sim-Eval → Deploy.
    All steps are fully automated (headless).
    """
    print("=" * 70)
    print("  QBank Pipeline: Full Generation (Headless)")
    print("=" * 70)

    # 1. Parse PDFs
    print("\n--- Phase 1/6: Parsing PDFs ---")
    pdf_parser.process_all_slides()

    # 2. Generate syllabi
    print("\n--- Phase 2/6: Generating Syllabi ---")
    syllabus_gen.process_all_syllabus()

    # 3. Generate MCQs (headless)
    print("\n--- Phase 3/6: Generating MCQs ---")
    mcq_files = mcq_gen.process_all_mcqs(headless=True)

    if not mcq_files:
        print("[Pipeline] No new MCQ files generated. Checking for existing files...")
        mcq_files = sorted(qa_config.GEN_DIR.glob("*.mcq_by_topic_*.json"))
        mcq_files = [f for f in mcq_files if ".shuffled." not in f.name]

    # 4. Shuffle options
    print("\n--- Phase 4/6: Shuffling Options ---")
    shuffled_files = []
    for mcq_file in mcq_files:
        shuffled = utils.shuffle_mcq_file(mcq_file)
        shuffled_files.append(shuffled)
        print(f"   Shuffled: {shuffled.name}")

    # 5. Simulated evaluation (headless)
    print("\n--- Phase 5/6: Cognitive Simulation Evaluation ---")
    evaluation.process_all_evaluations(headless=True)

    # 6. Deploy: copy to src/ and rebuild ChromaDB
    print("\n--- Phase 6/6: Deploying to Runtime ---")
    deployed = _deploy_qbank_files(shuffled_files if shuffled_files else mcq_files)

    if deployed:
        _rebuild_chroma_index(deployed)

    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)


def _deploy_qbank_files(source_files: List[Path]) -> List[Path]:
    """Copy generated QBank JSON files to the src/ runtime directory."""
    deployed = []
    for src_path in source_files:
        dest = SRC_DIR / src_path.name
        if src_path.resolve() == dest.resolve():
            print(f"   [Deploy] Already in place: {dest.name}")
            deployed.append(dest)
            continue
        shutil.copy2(src_path, dest)
        print(f"   [Deploy] Copied {src_path.name} → {dest}")
        deployed.append(dest)
    return deployed


# =====================
# Part B: ChromaDB Index Rebuild
# =====================

def _rebuild_chroma_index(qbank_paths: Optional[List[Path]] = None):
    """Rebuild ChromaDB vector index from QBank JSON files."""
    load_dotenv()
    oai = OpenAI()

    if qbank_paths is None:
        qbank_paths = [QBANK_JSON]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete and recreate collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids, all_docs, all_metas, all_embs = [], [], [], []

    for qbank_path in qbank_paths:
        if not qbank_path.exists():
            print(f"   [Reindex] Skipping {qbank_path.name} (not found)")
            continue

        data = json.loads(qbank_path.read_text(encoding="utf-8"))

        for topic in data.get("topics", []):
            parent_topic_id = topic.get("topic_id", "")
            for q in topic.get("questions", []):
                if q.get("status") == "deprecated":
                    continue

                qid = q.get("question_id", "")
                q_topic = q.get("topic_id", parent_topic_id)
                uid = f"{q_topic}:{qid}"
                diff = q.get("difficulty", 2)
                text_for_embed = build_qbank_search_text(q_topic, q)

                all_ids.append(uid)
                all_docs.append(text_for_embed)
                all_metas.append({
                    "topic_id": q_topic,
                    "question_id": qid,
                    "difficulty": diff,
                })

    # Batch embed
    print(f"   [Reindex] Embedding {len(all_ids)} active questions...")
    for start in range(0, len(all_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_ids))
        batch_texts = all_docs[start:end]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=batch_texts)
        for item in resp.data:
            all_embs.append(item.embedding)

    if all_ids:
        col.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            embeddings=all_embs,
        )

    print(f"   [Reindex] Done. Indexed {len(all_ids)} questions into '{COLLECTION_NAME}'.")


# =====================
# Part C: Quality Maintenance (replaces qbank_manager.py)
# =====================

def load_events(path: Path) -> List[dict]:
    """Load student event records from JSONL file."""
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def analyze_quality(events: List[dict]) -> List[QuestionQuality]:
    """
    Compute per-question quality metrics from real student data.
    Migrated from qbank_manager.py.
    """
    q_data: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {
        "total": 0, "correct": 0, "scores": []
    })

    for ev in events:
        tid = ev.get("topic_id", "")
        qid = ev.get("question_id", "")
        is_correct = ev.get("is_correct", False)
        rt = ev.get("response_time_sec", 0.0)
        d = q_data[(tid, qid)]
        d["total"] += 1
        if is_correct:
            d["correct"] += 1
        d["scores"].append((1 if is_correct else 0, rt))

    results: List[QuestionQuality] = []
    for (tid, qid), d in q_data.items():
        total = d["total"]
        correct = d["correct"]
        p = correct / total if total > 0 else 0.0

        scores = [s[0] for s in d["scores"]]
        if len(scores) >= 2:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            disc = min(1.0, (variance ** 0.5) * 2)
        else:
            disc = 0.0

        flag = "ok"
        if total >= MIN_ATTEMPTS_FOR_EVAL:
            if disc < DISCRIMINATION_THRESHOLD:
                flag = "low_discrimination"
            if p > DIFFICULTY_TOO_EASY:
                flag = "too_easy"
            elif p < DIFFICULTY_TOO_HARD:
                flag = "too_hard"

        results.append(QuestionQuality(
            topic_id=tid, question_id=qid,
            total_attempts=total, correct=correct,
            item_difficulty=round(p, 3),
            discrimination=round(disc, 3),
            flag=flag,
        ))

    results.sort(key=lambda r: r.discrimination)
    return results


def _evolve_flagged_question(
    oai: OpenAI,
    qbank: dict,
    quality: QuestionQuality,
) -> Optional[str]:
    """
    Use qbank_agent's mcq_gen to regenerate a flagged question.
    Returns new question_id if successful, None otherwise.
    """
    topic_id = quality.topic_id
    question_id = quality.question_id

    # Find the original question
    original_q = None
    target_topic = None
    for topic in qbank.get("topics", []):
        for q in topic.get("questions", []):
            if (q.get("question_id") == question_id and
                (q.get("topic_id") == topic_id or topic.get("topic_id") == topic_id)):
                original_q = q
                target_topic = topic
                break
        if original_q:
            break

    if not original_q:
        print(f"   [Evolve] Question {topic_id}:{question_id} not found, skipping.")
        return None

    # Use LLM to generate an improved version
    prompt = (
        "You are an expert curriculum designer. Regenerate this question to fix its quality issues.\n\n"
        f"**Original Question:**\n```json\n{json.dumps(original_q, ensure_ascii=False, indent=2)}\n```\n\n"
        f"**Quality Metrics:**\n"
        f"- Item Difficulty (p-value): {quality.item_difficulty}\n"
        f"- Discrimination: {quality.discrimination}\n"
        f"- Flag: {quality.flag}\n"
        f"- Total Attempts: {quality.total_attempts}\n\n"
        "**Requirements:**\n"
        "1. If 'low_discrimination': improve distractors to be plausible but clearly wrong\n"
        "2. If 'too_easy': require deeper conceptual understanding\n"
        "3. If 'too_hard': simplify wording or provide better context\n"
        "4. Keep the same knowledge point\n"
        "5. Maintain exactly four options A/B/C/D with one correct answer\n"
        "6. Output valid JSON with keys: stem, options (dict with A/B/C/D), answer, rationale, difficulty (1-3), knowledge_tags (list)\n"
    )

    try:
        resp = oai.chat.completions.create(
            model=qa_config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert curriculum designer. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        evolved_data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"   [Evolve] LLM call failed for {topic_id}:{question_id}: {e}")
        return None

    # Version control: deprecate old, append new
    new_qid = f"{question_id}_v{uuid.uuid4().hex[:6]}"

    original_q["status"] = "deprecated"
    original_q["replaced_by"] = new_qid

    new_q = {
        "question_id": new_qid,
        "topic_id": topic_id,
        "knowledge_tags": evolved_data.get("knowledge_tags", original_q.get("knowledge_tags", [])),
        "difficulty": evolved_data.get("difficulty", original_q.get("difficulty", 2)),
        "stem": evolved_data.get("stem", original_q.get("stem", "")),
        "options": evolved_data.get("options", original_q.get("options", {})),
        "answer": evolved_data.get("answer", original_q.get("answer", "")),
        "rationale": evolved_data.get("rationale", ""),
        "status": "active",
        "evolved_from": question_id,
    }
    target_topic["questions"].append(new_q)

    print(f"   [Evolve] {topic_id}:{question_id} → {new_qid} (flag={quality.flag})")
    return new_qid


def run_quality_maintenance():
    """
    Analyze real student data, auto-evolve flagged questions, rebuild index.
    This replaces qbank_manager.py's CLI commands.
    """
    print("=" * 70)
    print("  QBank Pipeline: Quality Maintenance")
    print("=" * 70)

    # 1. Load events
    events = load_events(EVENTS_FILE)
    if not events:
        print("[Maintain] No student event records found. Nothing to analyze.")
        return

    # 2. Analyze quality
    print(f"\n--- Analyzing {len(events)} student attempts ---")
    results = analyze_quality(events)

    flagged = [r for r in results if r.flag != "ok"]

    print(f"\n{'=' * 60}")
    print(f"Quality Report ({len(results)} questions with data)")
    print(f"{'=' * 60}")

    for r in results:
        marker = "🔴" if r.flag != "ok" else "🟢"
        print(
            f"  {marker} {r.topic_id}:{r.question_id}  "
            f"p={r.item_difficulty:.2f}  disc={r.discrimination:.2f}  "
            f"attempts={r.total_attempts}  flag={r.flag}"
        )

    if not flagged:
        print("\n✅ All questions passed quality checks. No evolution needed.")
        return

    # 3. Auto-evolve flagged questions
    print(f"\n⚠️  {len(flagged)} question(s) flagged. Auto-evolving...")

    load_dotenv()
    oai = OpenAI()

    if not QBANK_JSON.exists():
        print(f"[Error] QBank JSON not found: {QBANK_JSON}")
        return

    qbank = json.loads(QBANK_JSON.read_text(encoding="utf-8"))
    evolved_count = 0

    for quality in flagged:
        new_qid = _evolve_flagged_question(oai, qbank, quality)
        if new_qid:
            evolved_count += 1

    if evolved_count > 0:
        # 4. Save updated QBank
        QBANK_JSON.write_text(
            json.dumps(qbank, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n[Maintain] Saved {evolved_count} evolved question(s) to {QBANK_JSON.name}")

        # 5. Rebuild ChromaDB index
        print("\n--- Rebuilding ChromaDB Index ---")
        _rebuild_chroma_index([QBANK_JSON])

    print("\n" + "=" * 70)
    print(f"  Maintenance Complete! Evolved {evolved_count}/{len(flagged)} flagged questions.")
    print("=" * 70)


# =====================
# CLI Entry Point
# =====================

def main():
    parser = argparse.ArgumentParser(
        description="QBank Pipeline — 全自动题库生成、部署与维护"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Full pipeline: PDF → MCQ → Sim-Eval → Deploy")
    sub.add_parser("maintain", help="Analyze real student data → auto-evolve flagged questions")
    sub.add_parser("reindex", help="Rebuild ChromaDB index from current QBank JSON")

    args = parser.parse_args()

    if args.command == "generate":
        run_full_pipeline()
    elif args.command == "maintain":
        run_quality_maintenance()
    elif args.command == "reindex":
        load_dotenv()
        _rebuild_chroma_index()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
