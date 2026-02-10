import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field


# -----------------------
# Paths / Config
# -----------------------
CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"
QBANK_JSON = Path("Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_8.json")

SLIDE_FILE = Path("current_slide.txt")

EVENTS_FILE = Path("student_events.jsonl")      # episodic memory
PROFILE_FILE = Path("student_profile.json")     # semantic memory

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 12              # retrieve candidates (建议稍大一点，便于从中挑合适难度)
ASK_N_PER_SLIDE = 3     # 每次 slide change 问几题（课堂一般 1；你现在设 3 也可以）
WINDOW_SIZE = 5         # 每 5 题总结一次

DIFF_MIN, DIFF_MAX = 1, 3


# -----------------------
# Data models
# -----------------------
class Attempt(BaseModel):
    ts: float
    slide_text: str
    uid: str
    topic_id: str
    question_id: str
    difficulty: int = Field(ge=1, le=3)
    user_answer: str
    correct_answer: str
    is_correct: bool
    response_time_sec: float = Field(ge=0.0)


class DifficultySummary(BaseModel):
    performance: str = Field(description="One of: good / ok / weak")
    evidence: List[str] = Field(min_length=2, max_length=4)
    difficulty_delta: int = Field(description="One of -1, 0, +1")
    focus_topics: List[str] = Field(default_factory=list, description="Top topics to review or probe next")


class StudentProfile(BaseModel):
    target_difficulty: int = Field(ge=1, le=3, default=1)
    total_answered: int = Field(ge=0, default=0)
    summaries: List[DifficultySummary] = Field(default_factory=list)


# -----------------------
# Utilities
# -----------------------
def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def read_slide_text() -> str:
    if not SLIDE_FILE.exists():
        return ""
    return SLIDE_FILE.read_text(encoding="utf-8").strip()


def safe_load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path.resolve()} | error={e}")


def load_qbank_map(path: Path) -> Tuple[str, Dict[Tuple[str, str], dict], set]:
    data = safe_load_json(path)
    deck_id = data.get("deck_id", "unknown_deck")

    qmap: Dict[Tuple[str, str], dict] = {}
    topic_ids = set()

    for topic in data.get("topics", []):
        topic_id = topic.get("topic_id", "unknown_topic")
        topic_ids.add(topic_id)
        for q in topic.get("questions", []):
            qid = str(q.get("question_id", "unknown_q"))
            qmap[(topic_id, qid)] = q

    return deck_id, qmap, topic_ids


def embed_query(oai: OpenAI, text: str) -> List[float]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def append_event(attempt: Attempt) -> None:
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(attempt.model_dump_json() + "\n")


def load_last_attempts(n: int) -> List[Attempt]:
    if not EVENTS_FILE.exists():
        return []
    lines = EVENTS_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    tail = lines[-n:] if len(lines) >= n else lines
    attempts: List[Attempt] = []
    for line in tail:
        try:
            attempts.append(Attempt.model_validate_json(line))
        except Exception:
            continue
    return attempts


def load_profile() -> StudentProfile:
    if not PROFILE_FILE.exists():
        return StudentProfile(target_difficulty=1, total_answered=0, summaries=[])
    try:
        return StudentProfile.model_validate_json(PROFILE_FILE.read_text(encoding="utf-8"))
    except Exception:
        # 文件损坏/空文件时兜底
        return StudentProfile(target_difficulty=1, total_answered=0, summaries=[])


def save_profile(profile: StudentProfile) -> None:
    PROFILE_FILE.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def print_question(uid: str, topic_id: str, qid: str, q: dict) -> None:
    stem = (q.get("stem") or "").strip()
    options = q.get("options") or {}
    diff = q.get("difficulty", "")

    print("\n" + "=" * 80, flush=True)
    print(f"Ask: {uid} | topic_id={topic_id} | question_id={qid} | difficulty={diff}", flush=True)
    print(stem, flush=True)
    for k in ["A", "B", "C", "D"]:
        if k in options:
            print(f"  {k}. {options[k]}", flush=True)
    print("=" * 80, flush=True)


def pick_candidates_by_difficulty(
    ids: List[str],
    metas: List[dict],
    target_diff: int,
    max_pool: int,
) -> List[Tuple[str, dict]]:
    """
    在检索的候选池里优先按难度排序输出（不负责“问几题”，只负责返回排序好的候选列表）。
    """
    # 截断池大小
    ids = ids[:max_pool]
    metas = metas[:max_pool]

    buckets: Dict[int, List[Tuple[str, dict]]] = {1: [], 2: [], 3: []}
    for uid, meta in zip(ids, metas):
        try:
            d = int(meta.get("difficulty", 2))
        except Exception:
            d = 2
        d = clamp(d, DIFF_MIN, DIFF_MAX)
        buckets[d].append((uid, meta))

    ordered: List[Tuple[str, dict]] = []
    for d in [target_diff, target_diff - 1, target_diff + 1, 1, 2, 3]:
        if d < DIFF_MIN or d > DIFF_MAX:
            continue
        ordered.extend(buckets.get(d, []))

    # 去重保持顺序
    seen = set()
    uniq: List[Tuple[str, dict]] = []
    for uid, meta in ordered:
        if uid in seen:
            continue
        seen.add(uid)
        uniq.append((uid, meta))
    return uniq


# -----------------------
# LLM summarizer (every 5 attempts)
# -----------------------
def summarize_and_adjust(oai: OpenAI, profile: StudentProfile, window: List[Attempt]) -> DifficultySummary:
    total = len(window)
    correct = sum(1 for a in window if a.is_correct)
    acc = correct / total if total else 0.0

    by_diff = {
        1: {"total": 0, "correct": 0},
        2: {"total": 0, "correct": 0},
        3: {"total": 0, "correct": 0},
    }
    wrong_topics: Dict[str, int] = {}
    for a in window:
        d = clamp(int(a.difficulty), DIFF_MIN, DIFF_MAX)
        by_diff[d]["total"] += 1
        by_diff[d]["correct"] += 1 if a.is_correct else 0
        if not a.is_correct:
            wrong_topics[a.topic_id] = wrong_topics.get(a.topic_id, 0) + 1

    focus_topics = [t for t, _ in sorted(wrong_topics.items(), key=lambda x: x[1], reverse=True)[:3]]

    attempt_lines = []
    for a in window:
        attempt_lines.append(
            f"- {a.uid} diff={a.difficulty} user={a.user_answer} correct={a.correct_answer} "
            f"{'OK' if a.is_correct else 'WRONG'} time={a.response_time_sec:.1f}s topic={a.topic_id}"
        )

    system = (
        "You are a teaching assistant that summarizes a student's recent performance.\n"
        "You MUST use the provided numbers; do not invent counts.\n"
        "Output a JSON object matching the schema:\n"
        "- performance: 'good' or 'ok' or 'weak'\n"
        "- evidence: 2-4 short bullets, grounded in the window\n"
        "- difficulty_delta: -1 or 0 or +1\n"
        "- focus_topics: list of up to 3 topic_ids\n"
    )

    user = (
        f"Window size: {total}\n"
        f"Correct: {correct}/{total} (accuracy={acc:.2f})\n"
        f"Current target_difficulty: {profile.target_difficulty}\n"
        f"By difficulty: {by_diff}\n"
        f"Suggested focus_topics (from wrong answers): {focus_topics}\n"
        f"Attempts:\n" + "\n".join(attempt_lines) + "\n\n"
        "Now produce the structured DifficultySummary."
    )

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw_json = resp.choices[0].message.content
    summary = DifficultySummary.model_validate_json(raw_json)

    if summary.difficulty_delta not in (-1, 0, 1):
        summary.difficulty_delta = 0

    fixed_first = f"Last {total} questions: {correct}/{total} correct (accuracy={acc:.2f}), target_diff={profile.target_difficulty}."
    evidence_rest = [e for e in summary.evidence if isinstance(e, str)]
    summary.evidence = [fixed_first] + evidence_rest[:3]

    if not summary.focus_topics:
        summary.focus_topics = focus_topics

    return summary


# -----------------------
# Main loop
# -----------------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env or environment.")
    if not QBANK_JSON.exists():
        raise FileNotFoundError(f"Missing question bank JSON: {QBANK_JSON.resolve()}")

    deck_id, qmap, known_topic_ids = load_qbank_map(QBANK_JSON)
    print(f"Loaded qbank deck_id={deck_id}, total_questions={len(qmap)}", flush=True)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(name=COLLECTION_NAME)
    oai = OpenAI()

    profile = load_profile()
    print(f"Loaded profile: target_difficulty={profile.target_difficulty}, total_answered={profile.total_answered}", flush=True)

    last_slide: Optional[str] = None

    print(f"Watching {SLIDE_FILE.resolve()} ... (edit this file to simulate slide changes)", flush=True)
    print(f"Tip: you can write either natural language or a topic_id (e.g., {next(iter(known_topic_ids))})", flush=True)
    print("", flush=True)

    while True:
        slide_text = read_slide_text()

        if slide_text and slide_text != last_slide:
            last_slide = slide_text
            print(f"\n[Slide Update] {slide_text}", flush=True)

            # 如果 slide_text 刚好是 topic_id，直接按 topic_id 过滤（更稳定）
            where_filter = {"topic_id": slide_text} if slide_text in known_topic_ids else None

            q_emb = embed_query(oai, slide_text)

            try:
                res = col.query(
                    query_embeddings=[q_emb],
                    n_results=TOP_K,
                    include=["metadatas", "distances"],
                    where=where_filter,
                )
            except TypeError:
                # 某些 chromadb 版本 query() 可能不支持 where 参数
                res = col.query(
                    query_embeddings=[q_emb],
                    n_results=TOP_K,
                    include=["metadatas", "distances"],
                )

            ids = res["ids"][0]
            metas = res["metadatas"][0]
            dists = res.get("distances", [[None] * len(ids)])[0]

            if not ids:
                print("[Info] No retrieval results. If you used a topic_id, it may not exist in the Chroma index.", flush=True)
                print("[Info] If you recently changed the JSON file, make sure you rebuilt the Chroma index.", flush=True)
                time.sleep(1)
                continue

            # 打印检索到的前几个候选（便于你验证“换 slide 真的换题”）
            print("Top retrieved (first 5):", flush=True)
            for i, (uid, meta, dist) in enumerate(list(zip(ids, metas, dists))[:5], start=1):
                print(f"  [{i}] {uid} | topic={meta.get('topic_id')} qid={meta.get('question_id')} diff={meta.get('difficulty')} dist={dist}", flush=True)

            ordered_candidates = pick_candidates_by_difficulty(
                ids=ids,
                metas=metas,
                target_diff=profile.target_difficulty,
                max_pool=TOP_K,
            )

            asked_this_slide = 0
            skipped_not_found = 0

            for uid, meta in ordered_candidates:
                topic_id = meta.get("topic_id")
                qid = str(meta.get("question_id"))
                key = (topic_id, qid)

                if key not in qmap:
                    skipped_not_found += 1
                    continue

                q = qmap[key]
                print_question(uid, topic_id, qid, q)

                valid = [k for k in ["A", "B", "C", "D"] if k in (q.get("options") or {})]
                start = time.time()
                ans = input(f"Your answer ({'/'.join(valid)}): ").strip().upper()
                rt = time.time() - start

                correct = str(q.get("answer", "")).strip().upper()
                is_correct = (ans == correct)
                print(f"Your answer: {ans} | Correct: {correct} | {'OK' if is_correct else 'WRONG'}", flush=True)

                attempt = Attempt(
                    ts=time.time(),
                    slide_text=slide_text,
                    uid=uid,
                    topic_id=topic_id,
                    question_id=qid,
                    difficulty=clamp(int(q.get("difficulty", 2)), DIFF_MIN, DIFF_MAX),
                    user_answer=ans,
                    correct_answer=correct,
                    is_correct=is_correct,
                    response_time_sec=rt,
                )
                append_event(attempt)

                profile.total_answered += 1
                save_profile(profile)

                asked_this_slide += 1
                if asked_this_slide >= ASK_N_PER_SLIDE:
                    break

                # 每 WINDOW_SIZE 题总结一次（如果你 ASK_N_PER_SLIDE=3，切两次 slide 就可能触发）
                if profile.total_answered % WINDOW_SIZE == 0:
                    window = load_last_attempts(WINDOW_SIZE)
                    print("\n--- Summarizing last 5 attempts with LLM ---", flush=True)
                    summary = summarize_and_adjust(oai, profile, window)

                    new_target = clamp(profile.target_difficulty + summary.difficulty_delta, DIFF_MIN, DIFF_MAX)

                    print("Summary evidence:", flush=True)
                    for e in summary.evidence:
                        print(f"- {e}", flush=True)
                    print(f"Difficulty adjust: {profile.target_difficulty} -> {new_target} (delta={summary.difficulty_delta})", flush=True)
                    print(f"Focus topics: {summary.focus_topics}", flush=True)

                    profile.summaries.append(summary)
                    profile.target_difficulty = new_target
                    save_profile(profile)

            if asked_this_slide == 0:
                print("[Warn] Retrieved candidates, but none could be asked.", flush=True)
                if skipped_not_found > 0:
                    print("[Warn] Many retrieved items were not found in the current JSON map.", flush=True)
                    print("[Fix] This usually means your Chroma index was built from a different JSON file.", flush=True)
                    print("[Fix] Re-run your offline indexing script to rebuild the collection from the new JSON.", flush=True)

        time.sleep(1)


if __name__ == "__main__":
    main()
