import argparse
import json
import os
import sys
import time
import datetime
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from hybrid_retrieval import HybridRetriever

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

DATA_DIR = Path("data")
EVENTS_FILE = DATA_DIR / "student_events.jsonl"
PROFILE_FILE = DATA_DIR / "student_profile.json"
POPUP_QUEUE_FILE = Path("popup_data/popup_queue.jsonl")
POPUP_EVENTS_FILE = Path("popup_data/popup_events.jsonl")

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 12
ASK_N_PER_SLIDE = 3
WINDOW_SIZE = 5

DIFF_MIN, DIFF_MAX = 1, 3

POPUP_COOLDOWN_S = int(os.getenv("POPUP_COOLDOWN_S", str(8 * 60)))
POPUP_MAX_PER_HOUR = int(os.getenv("POPUP_MAX_PER_HOUR", "2"))
POPUP_HEUR_GATE = float(os.getenv("POPUP_HEUR_GATE", "0.60"))
POPUP_INTERACTIVE = os.getenv("POPUP_INTERACTIVE", "1").lower() in ("1", "true", "yes")

POPUP_MODEL = os.getenv("POPUP_MODEL", "openai:gpt-4o-mini")
GENERATOR_MODEL = "gpt-4o-mini"
GRADER_MODEL = "gpt-4o-mini"

# -----------------------
# Data models
# -----------------------
class Attempt(BaseModel):
    ts: float
    student_id: str = Field(default="default")
    session_date: str = Field(default="")
    slide_text: str
    uid: str
    topic_id: str
    question_id: str
    difficulty: int = Field(ge=1, le=3)
    user_answer: str
    correct_answer: str
    is_correct: bool
    confidence_score: float = Field(default=1.0)
    misconception_tags: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
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
    model_config = ConfigDict(extra="ignore")
    target_difficulty: int = Field(ge=1, le=3, default=1)
    total_answered: int = Field(ge=0, default=0)
    summaries: List[DifficultySummary] = Field(default_factory=list)
    topic_stats: Dict[str, TopicStats] = Field(default_factory=dict)
    popup_throttle: PopupThrottle = Field(default_factory=PopupThrottle)
    
class GradingResult(BaseModel):
    is_correct: bool = Field(description="Whether the student's answer is fundamentally correct.")
    confidence_score: float = Field(description="Confidence in this grading evaluation, from 0.0 to 1.0.")
    misconception_tags: List[str] = Field(default_factory=list, description="Specific misconceptions exhibited, if any.")
    reasoning: str = Field(description="Brief explanation of the grading decision.")

class TailoredQuestion(BaseModel):
    stem: str = Field(description="The tailored question stem.")
    option_a: str = Field(description="Text for option A.")
    option_b: str = Field(description="Text for option B.")
    option_c: str = Field(description="Text for option C.")
    option_d: str = Field(description="Text for option D.")
    answer: str = Field(description="The correct option key (e.g. 'A', 'B', 'C', 'D').")
    explanation: str = Field(description="Explanation of the correct answer.")

    @property
    def options(self) -> Dict[str, str]:
        return {"A": self.option_a, "B": self.option_b, "C": self.option_c, "D": self.option_d}

class PopupDecision(BaseModel):
    should_popup: bool = Field(description="Whether to show a check-in popup now.")
    reason: str = Field(description="Short rationale for logging/debug, not shown to student.")
    message: str = Field(description="Popup text shown to the student, keep it short.")
    cta: str = Field(description="One of: understood / not_sure / later")
    recommended_action: str = Field(description="One of: none / quick_check / practice_question")
    recommended_topic_id: Optional[str] = None

# ... (All previous helper classes for Tracker, Utils, Popup omitted for brevity, redefined below)
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
        if direction in (SlideDirection.FORWARD, SlideDirection.BACKWARD):
            self._dir_hist.append((now, direction))
        cutoff = now - 120.0
        self._dir_hist = [(t, d) for (t, d) in self._dir_hist if t >= cutoff]
        toggles = 0
        for i in range(1, len(self._dir_hist)):
            if self._dir_hist[i][1] != self._dir_hist[i - 1][1]:
                toggles += 1
        ev = TopicChangeEvent(
            ts=now, prev_topic=prev, curr_topic=curr_topic, dwell_prev_s=dwell_prev,
            direction=direction, span=span, toggles_120s=toggles,
        )
        self.last_topic = curr_topic
        self.last_change_ts = now
        return ev

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def read_slide_text() -> str:
    if not SLIDE_FILE.exists(): return ""
    return SLIDE_FILE.read_text(encoding="utf-8").strip()

def safe_load_json(path: Path) -> dict:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e: raise RuntimeError(f"Failed to parse JSON: {path.resolve()} | error={e}")

# -----------------------
# Session / Date helpers
# -----------------------
def get_current_date_fallback() -> str:
    """尝试获取网络时间作为当前日期 (YYYY-MM-DD)，失败则降级使用本地系统时间"""
    try:
        req = urllib.request.Request("http://worldtimeapi.org/api/timezone/Etc/UTC", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode('utf-8'))
            dt = datetime.datetime.fromisoformat(data["datetime"].replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"[Warning] Failed to fetch network time, using local time: {e}")
        return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")

def load_qbank_map(path: Path) -> Tuple[str, Dict[Tuple[str, str], dict], set, Dict[str, int]]:
    data = safe_load_json(path)
    deck_id = data.get("deck_id", "unknown_deck")
    qmap: Dict[Tuple[str, str], dict] = {}
    topic_ids = set()
    topic_order: Dict[str, int] = {}
    for idx, topic in enumerate(data.get("topics", []) or []):
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
    if not EVENTS_FILE.exists(): return []
    lines = EVENTS_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines: return []
    tail = lines[-n:] if len(lines) >= n else lines
    attempts: List[Attempt] = []
    for line in tail:
        try: attempts.append(Attempt.model_validate_json(line))
        except Exception: continue
    return attempts

def load_profile() -> StudentProfile:
    if not PROFILE_FILE.exists(): return StudentProfile()
    try: return StudentProfile.model_validate_json(PROFILE_FILE.read_text(encoding="utf-8"))
    except Exception: return StudentProfile()

def save_profile(profile: StudentProfile) -> None:
    PROFILE_FILE.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

def pick_candidates_by_difficulty(ids: List[str], metas: List[dict], target_diff: int, max_pool: int) -> List[Tuple[str, dict]]:
    ids = ids[:max_pool]
    metas = metas[:max_pool]
    buckets: Dict[int, List[Tuple[str, dict]]] = {1: [], 2: [], 3: []}
    for uid, meta in zip(ids, metas):
        try: d = int(meta.get("difficulty", 2))
        except Exception: d = 2
        d = clamp(d, DIFF_MIN, DIFF_MAX)
        buckets[d].append((uid, meta))
    ordered: List[Tuple[str, dict]] = []
    for d in [target_diff, target_diff - 1, target_diff + 1, 1, 2, 3]:
        if d < DIFF_MIN or d > DIFF_MAX: continue
        ordered.extend(buckets.get(d, []))
    seen = set()
    uniq: List[Tuple[str, dict]] = []
    for uid, meta in ordered:
        if uid in seen: continue
        seen.add(uid)
        uniq.append((uid, meta))
    return uniq

def update_topic_mastery(profile: StudentProfile, topic_id: str, is_correct: bool, alpha: float = 0.15) -> None:
    st = profile.topic_stats.get(topic_id) or TopicStats()
    st.n_attempted += 1
    if is_correct: st.n_correct += 1
    target = 1.0 if is_correct else 0.0
    st.ema_acc = (1.0 - alpha) * float(st.ema_acc) + alpha * target
    st.last_answer_ts = time.time()
    profile.topic_stats[topic_id] = st

def mastery_snapshot(profile: StudentProfile, topic_id: str) -> Dict[str, Any]:
    st = profile.topic_stats.get(topic_id) or TopicStats()
    acc = (st.n_correct / st.n_attempted) if st.n_attempted > 0 else None
    return {
        "topic_id": topic_id, "n_attempted": st.n_attempted, "n_correct": st.n_correct,
        "acc": acc, "ema_acc": float(st.ema_acc), "last_answer_ts": float(st.last_answer_ts),
    }

def throttle_allows(profile: StudentProfile, topic_id: str) -> Tuple[bool, str]:
    now = time.time()
    last = float(profile.popup_throttle.last_popup_ts_by_topic.get(topic_id, 0.0))
    if now - last < POPUP_COOLDOWN_S:
        remaining = int(POPUP_COOLDOWN_S - (now - last))
        return False, f"cooldown_active (remaining={remaining}s)"
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

def compute_confusion_features(ev: TopicChangeEvent, snap: Dict[str, Any]) -> Dict[str, Any]:
    ema = float(snap.get("ema_acc", 0.5))
    n_attempted = int(snap.get("n_attempted", 0))
    score = 0.45 + min(0.25, 0.05 * ev.span)
    if ev.dwell_prev_s < 60.0: score += 0.15
    if ev.toggles_120s >= 2: score += 0.15
    if n_attempted >= 2 and ema < 0.60: score += 0.25
    elif n_attempted == 0: score += 0.10
    score = max(0.0, min(1.0, score))
    return {
        "prev_topic": ev.prev_topic, "curr_topic": ev.curr_topic, "direction": ev.direction.value,
        "backtrack_span": ev.span, "toggles_120s": ev.toggles_120s, "dwell_prev_s": round(ev.dwell_prev_s, 2),
        "topic_n_attempted": n_attempted, "topic_ema_acc": round(ema, 3), "heuristic_confusion_score": round(score, 3),
    }

def summarize_and_adjust(oai: OpenAI, profile: StudentProfile, window: List[Attempt]) -> DifficultySummary:
    total = len(window)
    correct = sum(1 for a in window if a.is_correct)
    acc = correct / total if total else 0.0
    by_diff = {1: {"total": 0, "correct": 0}, 2: {"total": 0, "correct": 0}, 3: {"total": 0, "correct": 0}}
    wrong_topics: Dict[str, int] = {}
    for a in window:
        d = clamp(int(a.difficulty), DIFF_MIN, DIFF_MAX)
        by_diff[d]["total"] += 1
        by_diff[d]["correct"] += 1 if a.is_correct else 0
        if not a.is_correct: wrong_topics[a.topic_id] = wrong_topics.get(a.topic_id, 0) + 1

    focus_topics = [t for t, _ in sorted(wrong_topics.items(), key=lambda x: x[1], reverse=True)[:3]]
    attempt_lines = [
        f"- {a.uid} diff={a.difficulty} user={a.user_answer} correct={a.correct_answer} "
        f"{'OK' if a.is_correct else 'WRONG'} time={a.response_time_sec:.1f}s topic={a.topic_id}"
        for a in window
    ]

    try:
        resp = oai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a teaching assistant summarizing student performance."},
                {"role": "user", "content": f"Window size: {total}\nCorrect: {correct}/{total} (accuracy={acc:.2f})\nCurrent target_difficulty: {profile.target_difficulty}\nBy difficulty: {by_diff}\nFocus topics: {focus_topics}\nAttempts:\n" + "\n".join(attempt_lines)}
            ],
            response_format=DifficultySummary,
            temperature=0.2,
        )
        summary = resp.choices[0].message.parsed
        if not summary:
            raise ValueError("No parsed output")
    except Exception as e:
        print("[Warning] fallback for summary due to parse err:", e)
        summary = DifficultySummary(performance="ok", evidence=["Performance logged"], difficulty_delta=0, focus_topics=focus_topics)

    if summary.difficulty_delta not in (-1, 0, 1): summary.difficulty_delta = 0
    fixed_first = f"Last {total} questions: {correct}/{total} correct (accuracy={acc:.2f}), target_diff={profile.target_difficulty}."
    summary.evidence = [fixed_first] + summary.evidence[:3]
    if not summary.focus_topics: summary.focus_topics = focus_topics
    return summary

def infer_topic_id(slide_text: str, known_topic_ids: set, metas: List[dict]) -> str:
    s = (slide_text or "").strip()
    if s in known_topic_ids: return s
    if metas:
        t = metas[0].get("topic_id")
        if t: return str(t)
    return "unknown_topic"

def build_popup_agent() -> Any:
    if Agent is None: return None
    system_prompt = "You are an on-slide learning assistant. You decide whether to interrupt the student with a short, neutral check-in popup."
    instructions = "1) Default to NOT interrupting unless signals strongly suggest confusion.\n2) Provide exactly three options: understood / not_sure / later.\n"
    return Agent(POPUP_MODEL, output_type=PopupDecision, system_prompt=system_prompt, instructions=instructions, retries=1)

# -----------------------
# New LLM Features (Agentic RAG & Grader)
# -----------------------
def generate_tailored_question(oai: OpenAI, original_q: dict, focus_topics: List[str], target_diff: int, misconceptions: List[str], bridge_context: str = "") -> TailoredQuestion:
    # 强制动态降维: 当存在关联旧伤回忆时，将目标难度降到底部帮助复习
    if bridge_context.strip():
        target_diff = 1

    prompt = (
        f"Original Question Data: {json.dumps(original_q, ensure_ascii=False)}\n"
        f"Target Difficulty: {target_diff} (1=Easy, 3=Hard)\n"
    )

    if bridge_context.strip():
        prompt += (
            f"\n【高度关注：学生在相关概念上的历史薄弱点/错因（通过RAG向量检索匹配所得）】\n"
            f"{bridge_context}\n\n"
            "【出题要求 - 概念桥接(Concept Bridging)】\n"
            "系统判断上述历史薄弱点可能阻碍学生学习当前的话题。请你在生成本次定制题时：\n"
            "1. 务必将当前题目的难度下调到最易懂的基础档次。\n"
            "2. 在题干背景、选项、或解析中，显式地去修复、温习上述历史错误。\n"
            "3. 把它设计成一道用旧概念引导推导当前新概念的【概念桥接题】。\n"
            "确保有且仅有一个明确的正确答案。\n"
        )
    else:
        prompt += (
            f"Student Focus Topics (weaknesses): {focus_topics}\n"
            f"Recent Misconceptions: {misconceptions}\n\n"
            "Generate a modified, tailored question stem and distractors (choices) to specifically target the student's weaknesses while keeping it relevant to the topic.\n"
            "Ensure there is exactly one unquestionably correct option.\n"
        )
    
    try:
        resp = oai.beta.chat.completions.parse(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert tutor creating Agentic RAG tailored questions."},
                {"role": "user", "content": prompt}
            ],
            response_format=TailoredQuestion,
            temperature=0.7,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        print(f"[Error generating tailored question, returning original] {e}")
        opts = original_q.get("options", {})
        ans = original_q.get("answer", "")
        return TailoredQuestion(
            stem=original_q.get("stem", ""),
            option_a=opts.get("A", ""),
            option_b=opts.get("B", ""),
            option_c=opts.get("C", ""),
            option_d=opts.get("D", ""),
            answer=ans,
            explanation=original_q.get("explanation", "")
        )

def grade_answer(oai: OpenAI, question: TailoredQuestion, user_answer_text: str) -> GradingResult:
    prompt = (
        f"Question Stem: {question.stem}\n"
        f"Options: {json.dumps(question.options, ensure_ascii=False)}\n"
        f"Correct Answer Option: {question.answer}\n"
        f"Explanation: {question.explanation}\n\n"
        f"Student's Answer input: '{user_answer_text}'\n\n"
        "Evaluate the student's answer. If it aligns with the correct option or demonstrates correct reasoning, mark is_correct=true.\n"
        "Identify any key misconceptions if the answer is completely wrong."
    )
    try:
        resp = oai.beta.chat.completions.parse(
            model=GRADER_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator grading student responses."},
                {"role": "user", "content": prompt}
            ],
            response_format=GradingResult,
            temperature=0.1,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        print(f"[Error grading answer, falling back to basic matching] {e}")
        ans_clean = user_answer_text.strip().upper()
        corr_clean = str(question.answer).strip().upper()
        is_cor = (ans_clean == corr_clean)
        return GradingResult(is_correct=is_cor, confidence_score=0.5, misconception_tags=[], reasoning="Fallback exact match evaluation.")

# -----------------------
# OOP QuizSession
# -----------------------
class QuizSession:
    def __init__(self, student_id: str = "default", session_date: str = ""):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY")
        
        self.student_id = student_id
        self.session_date = session_date

        self.oai = OpenAI()
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.chroma.get_collection(name=COLLECTION_NAME)
        self.memory_col = self.chroma.get_or_create_collection(name=f"student_memory_{self.student_id}")
        self.qbank_retriever = HybridRetriever.from_qbank_json(QBANK_JSON, collection=self.col)
        self.memory_retriever = HybridRetriever.from_collection_snapshot(self.memory_col)
        
        self.deck_id, self.qmap, self.known_topic_ids, self.topic_order = load_qbank_map(QBANK_JSON)
        self.profile = load_profile()
        self.tracker = TopicTracker(topic_order=self.topic_order)
        self.popup_agent = build_popup_agent()
        
        self.last_slide: Optional[str] = None
        self.last_topic: Optional[str] = None
        
        print(f"QuizSession initialized. Student: {self.student_id} | Date: {self.session_date} | Deck: {self.deck_id} | Target Difficulty: {self.profile.target_difficulty}")
        
    def check_slide(self, slide_text: str) -> List[dict]:
        """
        Poll slide text and return questions to ask if slide changed.
        """
        if not slide_text or slide_text == self.last_slide:
            return []
            
        self.last_slide = slide_text
        print(f"\n[Slide Update] {slide_text}", flush=True)

        where_filter = {"topic_id": slide_text} if slide_text in self.known_topic_ids else None
        q_emb = embed_query(self.oai, slide_text)

        res = self.qbank_retriever.query(
            query_text=slide_text,
            query_embedding=q_emb,
            n_results=TOP_K,
            where_filter=where_filter,
        )

        ids = res.ids
        metas = res.metadatas
        if not ids: return []

        curr_topic = infer_topic_id(slide_text, self.known_topic_ids, metas)
        if curr_topic != self.last_topic:
            ev = self.tracker.on_topic_change(curr_topic)
            self.last_topic = curr_topic
            self._evaluate_popup(ev, curr_topic)

        # Reorder and tailor questions
        ordered_candidates = pick_candidates_by_difficulty(ids, metas, self.profile.target_difficulty, TOP_K)
        
        questions_to_ask = []
        for uid, meta in ordered_candidates:
            if len(questions_to_ask) >= ASK_N_PER_SLIDE: break
            
            topic_id = meta.get("topic_id")
            qid = str(meta.get("question_id"))
            key = (topic_id, qid)

            if key not in self.qmap: continue
            
            original_q = self.qmap[key]
            
            # Agentic RAG: Generate Tailored Variant
            recent_misconceptions = []
            if len(self.profile.summaries) > 0 and self.profile.summaries[-1].focus_topics:
                 # Extract some state info, currently just from latest focus_topics but ideally from episodic memory 
                 recent_misconceptions = ["Issues with topics: " + ", ".join(self.profile.summaries[-1].focus_topics)]

            # [New] Hybrid RAG retrieval: find semantically and lexically related historical weaknesses.
            bridge_context = ""
            try:
                mem_res = self.memory_retriever.query(
                    query_text=slide_text,
                    query_embedding=q_emb,
                    n_results=2,
                )
                
                valid_memories = []
                for doc, dist, hybrid_score in zip(
                    mem_res.documents,
                    mem_res.vector_distances,
                    mem_res.hybrid_scores,
                ):
                    if not doc:
                        continue
                    if (dist is not None and dist < 1.2) or hybrid_score >= 0.55:
                        valid_memories.append(doc)
                
                if valid_memories:
                    bridge_context = "\n".join(f"- {m}" for m in valid_memories)
                    print(f"\n[RAG Memory Retrieval] Found {len(valid_memories)} relevant historical weaknesses!")
            except Exception as e:
                # the collection might be empty initially
                pass
                
            tailored: TailoredQuestion = generate_tailored_question(
                self.oai, original_q, 
                focus_topics=self.profile.summaries[-1].focus_topics if self.profile.summaries else [],
                target_diff=self.profile.target_difficulty,
                misconceptions=recent_misconceptions,
                bridge_context=bridge_context
            )

            # Package info to return to simulator or interactive loop
            questions_to_ask.append({
                "uid": uid,
                "topic_id": topic_id,
                "question_id": qid,
                "original": original_q,
                "tailored": tailored
            })
            
        return questions_to_ask

    def process_answer(self, slide_text: str, uid: str, topic_id: str, qid: str, diff: int, user_answer: str, tailored: TailoredQuestion, response_time_sec: float):
        """
        Process the answer: exact-match first, then LLM grader if needed.
        Always prints the correct answer after evaluation.
        """
        # --- Hybrid grading: exact letter match check first ---
        user_clean = user_answer.strip().upper()
        correct_clean = tailored.answer.strip().upper()
        # Extract just the leading letter if user typed e.g. "A. because..."
        user_letter = user_clean[0] if user_clean and user_clean[0].isalpha() else user_clean

        if user_letter == correct_clean:
            eval_res = GradingResult(
                is_correct=True,
                confidence_score=1.0,
                misconception_tags=[],
                reasoning="Exact match."
            )
            print(f"[Grader] is_correct=True | confidence=1.00 (exact match)")
        else:
            eval_res: GradingResult = grade_answer(self.oai, tailored, user_answer)
            print(f"[Grader] is_correct={eval_res.is_correct} | confidence={eval_res.confidence_score:.2f}")
        
        if eval_res.reasoning and eval_res.reasoning != "Exact match.":
            print(f"[Grader Reasoning] {eval_res.reasoning}")
        if eval_res.misconception_tags:
            print(f"[Grader Misconceptions identified] {eval_res.misconception_tags}")

        # Always show the correct answer
        correct_text = tailored.options.get(correct_clean, "")
        print(f"[Correct Answer] {correct_clean}. {correct_text}")

        attempt = Attempt(
            ts=time.time(),
            student_id=self.student_id,
            session_date=self.session_date,
            slide_text=slide_text,
            uid=uid,
            topic_id=topic_id,
            question_id=qid,
            difficulty=clamp(diff, DIFF_MIN, DIFF_MAX),
            user_answer=user_answer,
            correct_answer=tailored.answer,
            is_correct=eval_res.is_correct,
            confidence_score=eval_res.confidence_score,
            misconception_tags=eval_res.misconception_tags,
            reasoning=eval_res.reasoning,
            response_time_sec=response_time_sec,
        )
        append_event(attempt)
        update_topic_mastery(self.profile, topic_id, eval_res.is_correct)
        
        # [New] Commit meaningful errors to the RAG memory for future cross-topic bridging
        if not eval_res.is_correct and (eval_res.misconception_tags or eval_res.reasoning):
            memo_text = f"Student failed question about '{slide_text}' (Topic: {topic_id}). "
            if eval_res.misconception_tags:
                memo_text += f"Identified misconceptions: {', '.join(eval_res.misconception_tags)}. "
            memo_text += f"Grader reasoning: {eval_res.reasoning}"
            try:
                memory_uid = f"mem_{uid}_{int(time.time())}"
                memory_meta = {"topic_id": topic_id, "date": self.session_date}
                mem_emb = embed_query(self.oai, memo_text)
                self.memory_col.add(
                    ids=[memory_uid],
                    embeddings=[mem_emb],
                    documents=[memo_text],
                    metadatas=[memory_meta]
                )
                self.memory_retriever.add_entry(memory_uid, memo_text, memory_meta)
            except Exception as e:
                print(f"[Warning] Failed to commit to RAG memory: {e}")
        
        self.profile.total_answered += 1
        if self.profile.total_answered % WINDOW_SIZE == 0:
            window = load_last_attempts(WINDOW_SIZE)
            summary = summarize_and_adjust(self.oai, self.profile, window)
            new_target = clamp(self.profile.target_difficulty + summary.difficulty_delta, DIFF_MIN, DIFF_MAX)
            self.profile.summaries.append(summary)
            self.profile.target_difficulty = new_target
            print(f"\n[Summary] Adapted difficulty -> {new_target}")

        save_profile(self.profile)
        return eval_res

    def _evaluate_popup(self, ev: Optional[TopicChangeEvent], curr_topic: str):
        if ev is None or ev.direction != SlideDirection.BACKWARD: return
        snap = mastery_snapshot(self.profile, curr_topic)
        feats = compute_confusion_features(ev, snap)
        allow, reason = throttle_allows(self.profile, curr_topic)
        if not allow: return
        heur = float(feats["heuristic_confusion_score"])
        if heur < POPUP_HEUR_GATE: return
        
        if not self.popup_agent: return
        
        prompt = (
            f"Topic change event: direction={ev.direction.value}, backtrack_span={ev.span}\n"
            f"toggles_120s={ev.toggles_120s}, dwell_prev_s={ev.dwell_prev_s:.2f}\n"
            f"Mastery snapshot for curr_topic:\n{json.dumps(snap, ensure_ascii=False)}\n"
            f"Computed features:\n{json.dumps(feats, ensure_ascii=False)}\nReturn PopupDecision."
        )
        try:
            rr = self.popup_agent.run_sync(prompt)
            decision: PopupDecision = rr.output
        except Exception: return

        if decision.should_popup:
            throttle_commit(self.profile, curr_topic)
            save_profile(self.profile)
            emit_popup_queue({"ts": ev.ts, "topic_id": curr_topic, "decision": decision.model_dump(), "features": feats})
            print(f"\n--- POPUP --- {decision.message}")
            if POPUP_INTERACTIVE:
                # Clear standard input buffer so buffered enters don't skip the prompt
                try:
                    import termios
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                except Exception:
                    pass
                print(f"(About previous topic: {curr_topic})")
                choice = input("Your choice 1:understood, 2:not_sure, 3:later -> ").strip()
                cta = "understood" if choice == "1" else "not_sure" if choice == "2" else "later"
                append_popup_event({"ts": time.time(), "topic_id": curr_topic, "cta": cta, "agent_cta": decision.cta, "features": feats})
                print(f"[Noted. Transitioning to new slide content...]\n")
                time.sleep(1.5)

    def generate_session_summary(self):
        """Generate an LLM-based summary of the student's performance this session."""
        # Collect this session's events
        events_file = DATA_DIR / "student_events.jsonl"
        if not events_file.exists():
            print("[Summary] No event records found.")
            return

        session_events = []
        for line in events_file.read_text(encoding="utf-8").strip().splitlines():
            try:
                ev = json.loads(line)
                if (ev.get("student_id") == self.student_id and
                    ev.get("session_date") == self.session_date):
                    session_events.append(ev)
            except Exception:
                continue

        if not session_events:
            print("[Summary] No events from this session to summarize.")
            return

        total = len(session_events)
        correct = sum(1 for e in session_events if e.get("is_correct"))
        accuracy = correct / total if total > 0 else 0

        # Build a concise record for the LLM
        records = []
        for e in session_events:
            records.append({
                "topic": e.get("topic_id", ""),
                "question": e.get("question_id", ""),
                "difficulty": e.get("difficulty", ""),
                "student_answer": e.get("user_answer", ""),
                "correct_answer": e.get("correct_answer", ""),
                "is_correct": e.get("is_correct", False),
                "misconceptions": e.get("misconception_tags", []),
                "response_time_sec": round(e.get("response_time_sec", 0), 1),
            })

        prompt = (
            f"你是一位教学助手。请根据以下学生本次课堂的答题记录，生成一份简洁的听课情况总结报告。\n\n"
            f"**学生 ID**: {self.student_id}\n"
            f"**日期**: {self.session_date}\n"
            f"**总答题数**: {total}\n"
            f"**正确率**: {accuracy:.0%}\n\n"
            f"**详细答题记录**:\n```json\n{json.dumps(records, ensure_ascii=False, indent=2)}\n```\n\n"
            f"请包含以下内容：\n"
            f"1. 总体表现评价（一句话）\n"
            f"2. 掌握较好的知识点\n"
            f"3. 需要加强的知识点和具体误解\n"
            f"4. 给学生的学习建议\n\n"
            f"使用中文输出，语气鼓励但诚实。控制在 300 字以内。"
        )

        try:
            resp = self.oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一位善于鼓励学生的教学助手，擅长从答题数据中提炼学习建议。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            summary_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Summary] LLM 调用失败: {e}")
            return

        # Print to terminal
        print("\n" + "=" * 70)
        print("  📊 本次听课情况总结")
        print("=" * 70)
        print(summary_text)
        print("=" * 70)

        # Save to file
        summaries_dir = DATA_DIR / "session_summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summaries_dir / f"{self.student_id}_{self.session_date}.md"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# 听课总结 — {self.student_id} ({self.session_date})\n\n")
            f.write(f"- 总答题数: {total}\n")
            f.write(f"- 正确率: {accuracy:.0%} ({correct}/{total})\n\n")
            f.write(summary_text + "\n")

        print(f"\n[Summary] 报告已保存至: {summary_path}")

    def on_session_end(self):
        """Triggered when the student session ends. Generates summary then runs quality maintenance."""
        # Step 1: Generate session summary for the student
        print("\n[Session End] 正在生成本次听课总结...")
        try:
            self.generate_session_summary()
        except Exception as e:
            print(f"[Session End] 总结生成失败: {e}")

        # Step 2: Run quality maintenance
        print("\n[Session End] Running automatic QBank quality maintenance...")
        try:
            from qbank_pipeline import run_quality_maintenance
            run_quality_maintenance()
        except Exception as e:
            print(f"[Session End] Quality maintenance skipped: {e}")

def run_interactive(student_id: str = "default", session_date: str = ""):
    session = QuizSession(student_id=student_id, session_date=session_date)
    print(f"Watching for slide changes... Edit `current_slide.txt` to trigger questions.", flush=True)
    
    try:
        while True:
            slide_text = read_slide_text()
            q_to_ask = session.check_slide(slide_text)
            
            for q_data in q_to_ask:
                tailored: TailoredQuestion = q_data['tailored']
                uid, topic_id, qid = q_data['uid'], q_data['topic_id'], q_data['question_id']
                diff_level = q_data['original'].get("difficulty", 2)
                
                print("\n" + "=" * 80)
                print(f"[Tailored Question] Original QID={qid} | Difficulty={diff_level}")
                print(f"STEM: {tailored.stem}")
                for k in ["A", "B", "C", "D"]:
                    if k in tailored.options:
                        print(f"  {k}. {tailored.options[k]}")
                print("=" * 80)

                start = time.time()
                # Clear standard input buffer so buffered enters don't skip the prompt
                try:
                    import termios, sys
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                except Exception:
                    pass
                ans = input("Your answer: ").strip()
                rt = time.time() - start
                
                session.process_answer(slide_text, uid, topic_id, qid, diff_level, ans, tailored, rt)
                
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Session] Student exited.")
        session.on_session_end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student quiz client — watch slides and answer questions.")
    parser.add_argument("--student", type=str, default="default", help="Student ID (e.g. --student alice)")
    args = parser.parse_args()

    session_date = get_current_date_fallback()
    print(f"[Session] Student '{args.student}' | Date {session_date}")
    run_interactive(student_id=args.student, session_date=session_date)
