"""
generate_report.py  —  Reporter (Instructor Feedback Agent)

Usage:
    # Full cohort report (all students, all sessions)
    python generate_report.py

    # Filter by student
    python generate_report.py --student alice

    # Filter by student + specific session
    python generate_report.py --student alice --session 1

Output:
    instructor_report.md (or instructor_report_<student>_session<N>.md)
    instructor_report.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# ---------------------
# Config
# ---------------------
EVENTS_FILE = Path("data/student_events.jsonl")
PROFILE_FILE = Path("data/student_profile.json")
REPORT_DIR = Path("reports")
REPORT_MODEL = "gpt-4o-mini"


# ---------------------
# Data Models
# ---------------------
class TopicMetrics(BaseModel):
    topic_id: str
    total_attempts: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_response_time_sec: float = 0.0
    common_wrong_answers: Dict[str, int] = Field(default_factory=dict)
    question_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class CohortReport(BaseModel):
    generated_at: str
    student_id: Optional[str] = None
    session_count: Optional[int] = None
    total_attempts: int
    overall_accuracy: float
    topic_metrics: List[TopicMetrics]
    weakest_topics: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)


# ---------------------
# Core Logic
# ---------------------
def load_events(path: Path) -> List[dict]:
    if not path.exists():
        print(f"[Error] Events file not found: {path.resolve()}")
        sys.exit(1)
    events = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def filter_events(events: List[dict], student_id: Optional[str], session_count: Optional[int]) -> List[dict]:
    filtered = events
    if student_id is not None:
        filtered = [e for e in filtered if e.get("student_id") == student_id]
    if session_count is not None:
        filtered = [e for e in filtered if e.get("session_count") == session_count]
    return filtered


def compute_metrics(events: List[dict], student_id: Optional[str] = None, session_count: Optional[int] = None) -> CohortReport:
    topic_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0, "correct": 0, "response_times": [],
        "wrong_answers": defaultdict(int), "questions": defaultdict(lambda: {"total": 0, "correct": 0})
    })

    for ev in events:
        tid = ev.get("topic_id", "unknown")
        qid = ev.get("question_id", "unknown")
        is_correct = ev.get("is_correct", False)
        rt = ev.get("response_time_sec", 0.0)
        user_ans = ev.get("user_answer", "")

        d = topic_data[tid]
        d["total"] += 1
        if is_correct:
            d["correct"] += 1
        else:
            d["wrong_answers"][user_ans] += 1
        d["response_times"].append(rt)

        qd = d["questions"][qid]
        qd["total"] += 1
        if is_correct:
            qd["correct"] += 1

    total_all = sum(d["total"] for d in topic_data.values())
    correct_all = sum(d["correct"] for d in topic_data.values())

    metrics: List[TopicMetrics] = []
    for tid, d in topic_data.items():
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        avg_rt = sum(d["response_times"]) / len(d["response_times"]) if d["response_times"] else 0.0
        q_breakdown = {}
        for qid, qd in d["questions"].items():
            q_acc = qd["correct"] / qd["total"] if qd["total"] > 0 else 0.0
            q_breakdown[qid] = {"total": qd["total"], "correct": qd["correct"], "accuracy": round(q_acc, 2)}

        metrics.append(TopicMetrics(
            topic_id=tid,
            total_attempts=d["total"],
            correct=d["correct"],
            accuracy=round(acc, 3),
            avg_response_time_sec=round(avg_rt, 2),
            common_wrong_answers=dict(d["wrong_answers"]),
            question_breakdown=q_breakdown,
        ))

    # Sort ascending by accuracy (weakest first)
    metrics.sort(key=lambda m: m.accuracy)

    weakest = [m.topic_id for m in metrics if m.accuracy < 0.5]

    anomalies = []
    for m in metrics:
        for qid, qd in m.question_breakdown.items():
            if qd["total"] >= 3 and qd["accuracy"] < 0.3:
                anomalies.append(
                    f"[WARNING] Question {m.topic_id}:{qid} has only {qd['accuracy']*100:.0f}% accuracy "
                    f"across {qd['total']} attempts. Consider reviewing question quality."
                )

    overall_acc = correct_all / total_all if total_all > 0 else 0.0

    return CohortReport(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        student_id=student_id,
        session_count=session_count,
        total_attempts=total_all,
        overall_accuracy=round(overall_acc, 3),
        topic_metrics=metrics,
        weakest_topics=weakest,
        anomalies=anomalies,
    )


def generate_markdown_with_llm(oai: OpenAI, report: CohortReport) -> str:
    """Use LLM to turn hard metrics into a readable Markdown instructor report."""

    scope_desc = "All students, all sessions"
    if report.student_id and report.session_count is not None:
        scope_desc = f"Student: {report.student_id} | Session: {report.session_count}"
    elif report.student_id:
        scope_desc = f"Student: {report.student_id} | All sessions"
    elif report.session_count is not None:
        scope_desc = f"All students | Session: {report.session_count}"

    data_block = report.model_dump_json(indent=2)

    prompt = (
        f"You are a teaching assistant helping a university instructor analyze in-class quiz data.\n"
        f"Scope of this report: {scope_desc}\n\n"
        "Below is the aggregated statistics (JSON format). Generate a Markdown instructor report that includes:\n"
        "1. **Overview**: total attempts, overall accuracy.\n"
        "2. **Per-Topic Analysis**: accuracy, avg response time, and most common wrong answers for each topic.\n"
        "3. **Top 3 Weakest Topics**: recommend for review in the next lecture.\n"
        "4. **Anomaly Alerts**: if any question has abnormally low accuracy, highlight it in an alert box.\n"
        "5. **Action Items**: 2–3 concrete, actionable teaching recommendations.\n\n"
        "Write the report concisely and professionally in English.\n\n"
        f"Data:\n```json\n{data_block}\n```"
    )

    resp = oai.chat.completions.create(
        model=REPORT_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional teaching analytics assistant. Always respond in English."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return resp.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(description="Reporter — Generate instructor feedback report")
    parser.add_argument("--student", type=str, default=None, help="Filter by student ID (default: all students)")
    parser.add_argument("--session", type=int, default=None, help="Filter by session number (default: all sessions)")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[Error] Missing OPENAI_API_KEY")
        sys.exit(1)

    print("[Reporter] Loading student events...", flush=True)
    all_events = load_events(EVENTS_FILE)
    print(f"[Reporter] Total records found: {len(all_events)}", flush=True)

    events = filter_events(all_events, args.student, args.session)
    print(f"[Reporter] Records after filtering (student={args.student}, session={args.session}): {len(events)}", flush=True)
    if not events:
        print("[Reporter] No records match the filter. Exiting.")
        sys.exit(0)

    print("[Reporter] Computing metrics...", flush=True)
    report = compute_metrics(events, student_id=args.student, session_count=args.session)

    print("[Reporter] Calling LLM to generate report...", flush=True)
    oai = OpenAI()
    md = generate_markdown_with_llm(oai, report)

    # Build output filename
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.student or args.session is not None:
        parts = ["instructor_report"]
        if args.student:
            parts.append(args.student)
        if args.session is not None:
            parts.append(f"session{args.session}")
        report_output = REPORT_DIR / ("_".join(parts) + ".md")
    else:
        report_output = REPORT_DIR / "instructor_report.md"

    report_output.write_text(md, encoding="utf-8")
    print(f"[Reporter] Report saved: {report_output.resolve()}", flush=True)

    json_path = report_output.with_suffix(".json")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"[Reporter] JSON data saved: {json_path.resolve()}", flush=True)

    # Print report to terminal
    print("\n" + "=" * 80)
    print(md)
    print("=" * 80)


if __name__ == "__main__":
    main()
