import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from hybrid_retrieval import HybridRetriever

# PydanticAI (for popup decision)
try:
    from pydantic_ai import Agent
except Exception:
    Agent = None


# -----------------------
# Paths / Config
# -----------------------
CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"
QBANK_JSON = Path("Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_8.json")

SLIDE_FILE = Path("current_slide.txt")

EVENTS_FILE = Path("student_events.jsonl")          # episodic memory (attempts)
PROFILE_FILE = Path("student_profile.json")         # semantic memory (profile + mastery + popup throttle)
POPUP_QUEUE_FILE = Path("popup_queue.jsonl")        # for UI/plugin consumption (jsonl)
POPUP_EVENTS_FILE = Path("popup_events.jsonl")      # popup interactions in terminal (jsonl)

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 12              # retrieve candidates
ASK_N_PER_SLIDE = 3     # per slide change
WINDOW_SIZE = 5         # summarize every N attempts

DIFF_MIN, DIFF_MAX = 1, 3

# Popup policy knobs (can override via env)
POPUP_COOLDOWN_S = int(os.getenv("POPUP_COOLDOWN_S", str(8 * 60)))
POPUP_MAX_PER_HOUR = int(os.getenv("POPUP_MAX_PER_HOUR", "2"))
POPUP_HEUR_GATE = float(os.getenv("POPUP_HEUR_GATE", "0.60"))  # only call LLM if heur_score >= gate
POPUP_INTERACTIVE = os.getenv("POPUP_INTERACTIVE", "1").lower() in ("1", "true", "yes")

# PydanticAI model for popup decision (can override via env)
POPUP_MODEL = os.getenv("POPUP_MODEL", "openai:gpt-4o-mini")


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


class TopicStats(BaseModel):
    n_attempted: int = 0
    n_correct: int = 0
    ema_acc: float = 0.5
    last_answer_ts: float = 0.0


class PopupThrottle(BaseModel):
    last_popup_ts_by_topic: Dict[str, float] = Field(default_factory=dict)
    prompts_last_hour: List[float] = Field(default_factory=list)


class StudentProfile(BaseModel):
    """
    Backward compatible: old profile files with only (target_difficulty,total_answered,summaries) still load.
    """
    model_config = ConfigDict(extra="ignore")

    target_difficulty: int = Field(ge=1, le=3, default=1)
    total_answered: int = Field(ge=0, default=0)
    summaries: List[DifficultySummary] = Field(default_factory=list)

    # New: mastery + popup throttling state
    topic_stats: Dict[str, TopicStats] = Field(default_factory=dict)
    popup_throttle: PopupThrottle = Field(default_factory=PopupThrottle)


# -----------------------
# Popup decision model (PydanticAI output_type)
# -----------------------
class PopupDecision(BaseModel):
    should_popup: bool = Field(description="Whether to show a check-in popup now.")
    reason: str = Field(description="Short rationale for logging/debug, not shown to student.")
    message: str = Field(description="Popup text shown to the student, keep it short.")
    cta: str = Field(description="One of: understood / not_sure / later")
    recommended_action: str = Field(description="One of: none / quick_check / practice_question")
    recommended_topic_id: Optional[str] = None


# -----------------------
# Slide tracking
# -----------------------
class SlideDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class TopicChangeEvent:
    ts: float
    prev_topic: str
    curr_topic: str
    dwell_prev_s: float
    direction: SlideDirection
    span: int
    toggles_120s: int


@dataclass
class TopicTracker:
    """
    Track topic transitions (not raw slide text), using deck topic order when available.
    """
    topic_order: Dict[str, int]
    last_topic: Optional[str] = None
    last_change_ts: float = field(default_factory=lambda: time.time())

    _dir_hist: List[Tuple[float, SlideDirection]] = field(default_factory=list)

    def _idx(self, topic_id: str) -> int:
        if topic_id not in self.topic_order:
            self.topic_order[topic_id] = len(self.topic_order)
        return self.topic_order[topic_id]

    def on_topic_change(self, curr_topic: str, now: Optional[float] = None) -> Optional[TopicChangeEvent]:
        now = time.time() if now is None else now

        if self.last_topic is None:
            self._idx(curr_topic)
            self.last_topic = curr_topic
            self.last_change_ts = now
            return None

        prev = self.last_topic
        if curr_topic == prev:
            return None

        dwell_prev = max(0.0, now - self.last_change_ts)

        prev_idx = self._idx(prev)
        curr_idx = self._idx(curr_topic)

        if curr_idx > prev_idx:
            direction = SlideDirection.FORWARD
            span = curr_idx - prev_idx
        else:
            direction = SlideDirection.BACKWARD
            span = prev_idx - curr_idx

        # toggle detection within 120s
        if direction in (SlideDirection.FORWARD, SlideDirection.BACKWARD):
            self._dir_hist.append((now, direction))
        cutoff = now - 120.0
        self._dir_hist = [(t, d) for (t, d) in self._dir_hist if t >= cutoff]

        toggles = 0
        for i in range(1, len(self._dir_hist)):
            if self._dir_hist[i][1] != self._dir_hist[i - 1][1]:
                toggles += 1

        ev = TopicChangeEvent(
            ts=now,
            prev_topic=prev,
            curr_topic=curr_topic,
            dwell_prev_s=dwell_prev,
            direction=direction,
            span=span,
            toggles_120s=toggles,
        )

        self.last_topic = curr_topic
        self.last_change_ts = now
        return ev


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


def load_qbank_map(path: Path) -> Tuple[str, Dict[Tuple[str, str], dict], set, Dict[str, int]]:
    data = safe_load_json(path)
    deck_id = data.get("deck_id", "unknown_deck")

    qmap: Dict[Tuple[str, str], dict] = {}
    topic_ids = set()
    topic_order: Dict[str, int] = {}

    topics = data.get("topics", []) or []
    for idx, topic in enumerate(topics):
        topic_id = topic.get("topic_id", "unknown_topic")
        topic_ids.add(topic_id)
        topic_order[topic_id] = idx
        for q in topic.get("questions", []) or []:
            qid = str(q.get("question_id", "unknown_q"))
            qmap[(topic_id, qid)] = q

    return deck_id, qmap, topic_ids, topic_order


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
    In retrieval candidate pool, reorder by proximity to target difficulty.
    """
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

    seen = set()
    uniq: List[Tuple[str, dict]] = []
    for uid, meta in ordered:
        if uid in seen:
            continue
        seen.add(uid)
        uniq.append((uid, meta))
    return uniq


# -----------------------
# Mastery update + snapshots
# -----------------------
def update_topic_mastery(profile: StudentProfile, topic_id: str, is_correct: bool, alpha: float = 0.15) -> None:
    st = profile.topic_stats.get(topic_id) or TopicStats()
    st.n_attempted += 1
    if is_correct:
        st.n_correct += 1
    target = 1.0 if is_correct else 0.0
    st.ema_acc = (1.0 - alpha) * float(st.ema_acc) + alpha * target
    st.last_answer_ts = time.time()
    profile.topic_stats[topic_id] = st


def mastery_snapshot(profile: StudentProfile, topic_id: str) -> Dict[str, Any]:
    st = profile.topic_stats.get(topic_id) or TopicStats()
    acc = (st.n_correct / st.n_attempted) if st.n_attempted > 0 else None
    return {
        "topic_id": topic_id,
        "n_attempted": st.n_attempted,
        "n_correct": st.n_correct,
        "acc": acc,
        "ema_acc": float(st.ema_acc),
        "last_answer_ts": float(st.last_answer_ts),
    }


# -----------------------
# Popup throttle + logging
# -----------------------
def throttle_allows(profile: StudentProfile, topic_id: str) -> Tuple[bool, str]:
    now = time.time()

    # cooldown by topic
    last = float(profile.popup_throttle.last_popup_ts_by_topic.get(topic_id, 0.0))
    if now - last < POPUP_COOLDOWN_S:
        remaining = int(POPUP_COOLDOWN_S - (now - last))
        return False, f"cooldown_active (remaining={remaining}s)"

    # max per hour
    cutoff = now - 3600.0
    profile.popup_throttle.prompts_last_hour = [t for t in profile.popup_throttle.prompts_last_hour if t >= cutoff]
    if len(profile.popup_throttle.prompts_last_hour) >= POPUP_MAX_PER_HOUR:
        return False, "max_per_hour_reached"

    return True, "ok"


def throttle_commit(profile: StudentProfile, topic_id: str) -> None:
    now = time.time()
    profile.popup_throttle.last_popup_ts_by_topic[topic_id] = now
    profile.popup_throttle.prompts_last_hour.append(now)


def emit_popup_queue(payload: Dict[str, Any]) -> None:
    POPUP_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with POPUP_QUEUE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_popup_event(payload: Dict[str, Any]) -> None:
    POPUP_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with POPUP_EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------
# Popup heuristic features
# -----------------------
def compute_confusion_features(
    ev: TopicChangeEvent,
    snap: Dict[str, Any],
) -> Dict[str, Any]:
    ema = float(snap.get("ema_acc", 0.5))
    n_attempted = int(snap.get("n_attempted", 0))

    score = 0.0

    # backtrack is necessary condition (we only call this on BACKWARD)
    score += 0.45
    score += min(0.25, 0.05 * ev.span)

    # quick backtrack after short dwell on prev topic -> stronger signal
    if ev.dwell_prev_s < 60.0:
        score += 0.15

    # toggling forward/back in short window -> strong confusion signal
    if ev.toggles_120s >= 2:
        score += 0.15

    # mastery low -> stronger
    if n_attempted >= 2 and ema < 0.60:
        score += 0.25
    elif n_attempted == 0:
        score += 0.10

    score = max(0.0, min(1.0, score))

    return {
        "prev_topic": ev.prev_topic,
        "curr_topic": ev.curr_topic,
        "direction": ev.direction.value,
        "backtrack_span": ev.span,
        "toggles_120s": ev.toggles_120s,
        "dwell_prev_s": round(ev.dwell_prev_s, 2),
        "topic_n_attempted": n_attempted,
        "topic_ema_acc": round(ema, 3),
        "heuristic_confusion_score": round(score, 3),
    }


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
# Topic inference helper
# -----------------------
def infer_topic_id(slide_text: str, known_topic_ids: set, ids: List[str], metas: List[dict]) -> str:
    """
    Prefer:
    1) slide_text is exactly a topic_id
    2) else use top1 retrieval metadata topic_id
    """
    s = (slide_text or "").strip()
    if s in known_topic_ids:
        return s
    if metas:
        t = metas[0].get("topic_id")
        if t:
            return str(t)
    return "unknown_topic"


# -----------------------
# Popup agent init
# -----------------------
def build_popup_agent() -> Any:
    if Agent is None:
        raise RuntimeError(
            "Missing dependency: pydantic-ai. Install with:\n"
            "  python3 -m pip install -U pydantic-ai\n"
            "Then re-run."
        )

    system_prompt = (
        "You are an on-slide learning assistant.\n"
        "You decide whether to interrupt the student with a short, neutral check-in popup.\n"
    )
    instructions = (
        "Rules:\n"
        "1) Default to NOT interrupting unless signals strongly suggest confusion.\n"
        "2) If you choose to popup, keep message neutral and short.\n"
        "3) Provide exactly three options: understood / not_sure / later.\n"
        "4) Base the decision ONLY on the provided signals.\n"
        "5) Output must match PopupDecision schema.\n"
    )

    return Agent(
        POPUP_MODEL,
        output_type=PopupDecision,
        system_prompt=system_prompt,
        instructions=instructions,
        retries=1,
    )


# -----------------------
# Main loop
# -----------------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env or environment.")
    if not QBANK_JSON.exists():
        raise FileNotFoundError(f"Missing question bank JSON: {QBANK_JSON.resolve()}")

    deck_id, qmap, known_topic_ids, topic_order = load_qbank_map(QBANK_JSON)
    print(f"Loaded qbank deck_id={deck_id}, total_questions={len(qmap)}", flush=True)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(name=COLLECTION_NAME)
    retriever = HybridRetriever.from_qbank_json(QBANK_JSON, collection=col)
    oai = OpenAI()

    profile = load_profile()
    print(f"Loaded profile: target_difficulty={profile.target_difficulty}, total_answered={profile.total_answered}", flush=True)

    # topic tracker uses deck order, and will append unknown topics if any
    tracker = TopicTracker(topic_order=topic_order)

    popup_agent = build_popup_agent()

    last_slide: Optional[str] = None
    last_topic: Optional[str] = None

    print(f"Watching {SLIDE_FILE.resolve()} ... (edit this file to simulate slide changes)", flush=True)
    print(f"Tip: you can write either natural language or a topic_id (e.g., {next(iter(known_topic_ids))})", flush=True)
    print(f"Popup model: {POPUP_MODEL} | cooldown={POPUP_COOLDOWN_S}s | max_per_hour={POPUP_MAX_PER_HOUR} | heur_gate={POPUP_HEUR_GATE}", flush=True)
    print("", flush=True)

    while True:
        slide_text = read_slide_text()

        if slide_text and slide_text != last_slide:
            last_slide = slide_text
            print(f"\n[Slide Update] {slide_text}", flush=True)

            # If slide_text is topic_id, we can filter for stability
            where_filter = {"topic_id": slide_text} if slide_text in known_topic_ids else None

            q_emb = embed_query(oai, slide_text)
            res = retriever.query(
                query_text=slide_text,
                query_embedding=q_emb,
                n_results=TOP_K,
                where_filter=where_filter,
            )

            ids = res.ids
            metas = res.metadatas
            dists = res.vector_distances
            hybrid_scores = res.hybrid_scores
            bm25_scores = res.bm25_scores

            if not ids:
                print("[Info] No retrieval results. If you used a topic_id, it may not exist in the Chroma index.", flush=True)
                print("[Info] If you recently changed the JSON file, make sure you rebuilt the Chroma index.", flush=True)
                time.sleep(1)
                continue

            # Print top retrieved candidates for debug
            print("Top retrieved (first 5):", flush=True)
            for i, (uid, meta, dist, hybrid_score, bm25_score) in enumerate(
                list(zip(ids, metas, dists, hybrid_scores, bm25_scores))[:5],
                start=1,
            ):
                print(
                    f"  [{i}] {uid} | topic={meta.get('topic_id')} qid={meta.get('question_id')} "
                    f"diff={meta.get('difficulty')} hybrid={hybrid_score:.4f} "
                    f"vec_dist={dist} bm25={bm25_score:.4f}",
                    flush=True
                )

            # Infer topic_id for topic-level tracking
            curr_topic = infer_topic_id(slide_text, known_topic_ids, ids, metas)
            if curr_topic != last_topic:
                ev = tracker.on_topic_change(curr_topic)
                last_topic = curr_topic

                # Backtrack detection -> maybe popup
                if ev is not None and ev.direction == SlideDirection.BACKWARD:
                    snap = mastery_snapshot(profile, curr_topic)
                    feats = compute_confusion_features(ev, snap)

                    allow, reason = throttle_allows(profile, curr_topic)
                    if not allow:
                        print(f"[Popup Throttled] topic={curr_topic} reason={reason}", flush=True)
                    else:
                        heur = float(feats["heuristic_confusion_score"])
                        if heur < POPUP_HEUR_GATE:
                            print(f"[Popup Skipped] heur_score={heur:.3f} < gate={POPUP_HEUR_GATE}", flush=True)
                        else:
                            prompt = (
                                "Decide if we should show a check-in popup now.\n\n"
                                f"Topic change event:\n"
                                f"- prev_topic: {ev.prev_topic}\n"
                                f"- curr_topic: {ev.curr_topic}\n"
                                f"- direction: {ev.direction.value}\n"
                                f"- backtrack_span: {ev.span}\n"
                                f"- toggles_120s: {ev.toggles_120s}\n"
                                f"- dwell_prev_s: {ev.dwell_prev_s:.2f}\n\n"
                                f"Mastery snapshot for curr_topic:\n{json.dumps(snap, ensure_ascii=False)}\n\n"
                                f"Computed features:\n{json.dumps(feats, ensure_ascii=False)}\n\n"
                                "Return PopupDecision."
                            )

                            try:
                                rr = popup_agent.run_sync(prompt)
                                decision: PopupDecision = rr.output
                            except Exception as e:
                                print(f"[Popup Agent Error] {e}", flush=True)
                                decision = PopupDecision(
                                    should_popup=False,
                                    reason=f"agent_error: {e}",
                                    message="",
                                    cta="later",
                                    recommended_action="none",
                                    recommended_topic_id=None,
                                )

                            if decision.should_popup:
                                throttle_commit(profile, curr_topic)
                                save_profile(profile)

                                payload = {
                                    "ts": ev.ts,
                                    "topic_id": curr_topic,
                                    "decision": decision.model_dump(),
                                    "features": feats,
                                }
                                emit_popup_queue(payload)

                                print(f"[Popup Emitted] topic={curr_topic} message={decision.message!r}", flush=True)

                                if POPUP_INTERACTIVE:
                                    print("\n--- POPUP ---", flush=True)
                                    print(decision.message, flush=True)
                                    print("1) understood   2) not_sure   3) later", flush=True)
                                    choice = input("Your choice (1/2/3): ").strip()

                                    cta = "later"
                                    if choice == "1":
                                        cta = "understood"
                                    elif choice == "2":
                                        cta = "not_sure"
                                    elif choice == "3":
                                        cta = "later"

                                    append_popup_event(
                                        {
                                            "ts": time.time(),
                                            "topic_id": curr_topic,
                                            "cta": cta,
                                            "agent_cta": decision.cta,
                                            "recommended_action": decision.recommended_action,
                                            "reason": decision.reason,
                                            "features": feats,
                                        }
                                    )
                                    print("--- POPUP END ---\n", flush=True)
                            else:
                                print(f"[No Popup] reason={decision.reason}", flush=True)

            # Now do question selection for this slide, same as before
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

                # update mastery immediately
                update_topic_mastery(profile, topic_id, is_correct)

                profile.total_answered += 1

                # Summarize every WINDOW_SIZE attempts (fixed: do this BEFORE potential break)
                if profile.total_answered % WINDOW_SIZE == 0:
                    window = load_last_attempts(WINDOW_SIZE)
                    print(f"\n--- Summarizing last {WINDOW_SIZE} attempts with LLM ---", flush=True)
                    summary = summarize_and_adjust(oai, profile, window)

                    new_target = clamp(profile.target_difficulty + summary.difficulty_delta, DIFF_MIN, DIFF_MAX)

                    print("Summary evidence:", flush=True)
                    for e in summary.evidence:
                        print(f"- {e}", flush=True)
                    print(
                        f"Difficulty adjust: {profile.target_difficulty} -> {new_target} "
                        f"(delta={summary.difficulty_delta})",
                        flush=True
                    )
                    print(f"Focus topics: {summary.focus_topics}", flush=True)

                    profile.summaries.append(summary)
                    profile.target_difficulty = new_target

                # persist after each attempt
                save_profile(profile)

                asked_this_slide += 1
                if asked_this_slide >= ASK_N_PER_SLIDE:
                    break

            if asked_this_slide == 0:
                print("[Warn] Retrieved candidates, but none could be asked.", flush=True)
                if skipped_not_found > 0:
                    print("[Warn] Many retrieved items were not found in the current JSON map.", flush=True)
                    print("[Fix] This usually means your Chroma index was built from a different JSON file.", flush=True)
                    print("[Fix] Re-run your offline indexing script to rebuild the collection from the new JSON.", flush=True)

        time.sleep(1)


if __name__ == "__main__":
    main()
