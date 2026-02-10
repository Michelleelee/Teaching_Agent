import json
import random
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel


# -----------------------------
# Immediate debug prints (guaranteed)
# -----------------------------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
CWD = Path.cwd()

QBANK_FILE = SCRIPT_DIR / "Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic.json"

print("[run_mastery_quiz] Script imported.", flush=True)
print(f"[run_mastery_quiz] __file__ = {SCRIPT_PATH}", flush=True)
print(f"[run_mastery_quiz] cwd      = {CWD}", flush=True)
print(f"[run_mastery_quiz] qbank    = {QBANK_FILE} (exists={QBANK_FILE.exists()})", flush=True)


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "gpt-4o-mini"
NUM_QUESTIONS = 10
DELTA_MIN = -0.15
DELTA_MAX = 0.15


# -----------------------------
# Question bank models
# -----------------------------
class Question(BaseModel):
    uid: str  # topic_id:question_id
    deck_id: str
    topic_id: str
    knowledge_tags: List[str] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=3)
    stem: str
    options: Dict[str, str]
    answer: str  # "A"/"B"/"C"/"D"
    rationale: Optional[str] = None


class Attempt(BaseModel):
    uid: str
    topic_id: str
    difficulty: int
    knowledge_tags: List[str] = Field(default_factory=list)
    user_answer: str
    correct_answer: str
    is_correct: bool
    response_time_sec: float = Field(ge=0.0)


# -----------------------------
# Deps + Output (LLM calibration with free-text evidence)
# -----------------------------
class MasteryDeps(BaseModel):
    deck_id: str
    attempts: List[Attempt]
    correct_count: int = Field(ge=0)
    total_count: int = Field(ge=1)
    base_score: float = Field(ge=0.0, le=1.0)
    avg_difficulty: float = Field(ge=1.0, le=3.0)
    avg_time_sec: float = Field(ge=0.0)


class MasteryEstimate(BaseModel):
    mastery: float = Field(ge=0.0, le=1.0)
    calibration_delta: float = Field(ge=DELTA_MIN, le=DELTA_MAX)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(min_length=1, max_length=3)
    next_action: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def load_question_bank(path: Path) -> List[Question]:
    if not path.exists():
        raise FileNotFoundError(f"Question bank not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    deck_id = data.get("deck_id", "unknown_deck")
    questions: List[Question] = []

    for t in data.get("topics", []):
        topic_id = t.get("topic_id", "unknown_topic")
        for q in t.get("questions", []):
            qid = q.get("question_id", "unknown_q")
            uid = f"{topic_id}:{qid}"

            question = Question(
                uid=uid,
                deck_id=deck_id,
                topic_id=topic_id,
                knowledge_tags=q.get("knowledge_tags", []) or [],
                difficulty=int(q.get("difficulty", 2)),
                stem=(q.get("stem", "") or "").strip(),
                options=q.get("options", {}) or {},
                answer=str(q.get("answer", "")).strip().upper(),
                rationale=q.get("rationale"),
            )

            if question.stem and question.options and question.answer in question.options:
                questions.append(question)

    if not questions:
        raise ValueError("Loaded 0 valid questions. Please check JSON structure/fields.")

    return questions


def prompt_user_answer(valid_keys: List[str]) -> str:
    valid_set = {k.upper() for k in valid_keys}
    while True:
        ans = input(f"Your answer ({'/'.join(valid_keys)}), or 'q' to quit: ").strip().upper()
        if ans == "Q":
            return "Q"
        if ans in valid_set:
            return ans
        print(f"Invalid input. Please enter one of: {', '.join(valid_keys)} (or 'q' to quit).", flush=True)


def compute_stats(attempts: List[Attempt]) -> Tuple[int, int, float, float, float]:
    total = len(attempts)
    correct = sum(1 for a in attempts if a.is_correct)
    base = correct / total if total else 0.0
    avg_diff = sum(a.difficulty for a in attempts) / total if total else 2.0
    avg_time = sum(a.response_time_sec for a in attempts) / total if total else 0.0
    return correct, total, base, avg_diff, avg_time


def canonical_first_bullet(correct: int, total: int, base: float) -> str:
    return f"Correct={correct}/{total} (base_score={base:.2f})."


def sanitize_evidence(
    llm_lines: List[str],
    correct: int,
    total: int,
    base: float,
) -> Tuple[List[str], bool]:
    """
    Keep LLM free-text evidence, but auto-fix hallucinated counts:
    - 'x out of y' and 'x/y' rewritten to correct/total
    - 'y attempts/questions/items' rewritten to total
    - 'x correct/right' rewritten to correct (best-effort)
    Also forces the first bullet to be deterministic correct stats.
    """
    was_corrected = False
    fixed: List[str] = []

    pat_out_of = re.compile(r"(\d+)\s*out\s*of\s*(\d+)", re.IGNORECASE)
    pat_frac = re.compile(r"(\d+)\s*/\s*(\d+)")
    pat_attempts = re.compile(r"(\d+)\s*(attempts|questions|items)", re.IGNORECASE)
    pat_correct_word = re.compile(r"(\d+)\s*(correct|right)", re.IGNORECASE)

    for line in llm_lines:
        if not isinstance(line, str):
            was_corrected = True
            continue

        s = line.strip()
        orig = s

        # Replace "x out of y" -> "correct out of total"
        if pat_out_of.search(s):
            s = pat_out_of.sub(f"{correct} out of {total}", s)

        # Replace "x/y" -> "correct/total"
        if pat_frac.search(s):
            s = pat_frac.sub(f"{correct}/{total}", s)

        # Replace "y attempts/questions/items" -> total
        def repl_attempts(m: re.Match) -> str:
            nonlocal was_corrected
            n = int(m.group(1))
            word = m.group(2)
            if n != total:
                was_corrected = True
                return f"{total} {word}"
            return m.group(0)

        s = pat_attempts.sub(repl_attempts, s)

        # Replace "x correct/right" -> correct (best-effort)
        # Note: might also match phrases like "the correct answer", but pattern requires a leading number.
        def repl_correct(m: re.Match) -> str:
            nonlocal was_corrected
            n = int(m.group(1))
            word = m.group(2)
            if n != correct:
                was_corrected = True
                return f"{correct} {word}"
            return m.group(0)

        s = pat_correct_word.sub(repl_correct, s)

        if s != orig:
            was_corrected = True

        # We always inject the canonical first bullet ourselves; avoid duplicate.
        if "Correct=" in s:
            continue

        if s:
            fixed.append(s)

    # Final evidence: 1 canonical + up to 2 LLM free-text lines
    final = [canonical_first_bullet(correct, total, base)] + fixed[:2]
    return final, was_corrected


# -----------------------------
# Build Agent (LLM generates free-text evidence)
# -----------------------------
def build_mastery_agent() -> Agent:
    model = OpenAIChatModel(MODEL_NAME)

    agent = Agent(
        model=model,
        output_type=MasteryEstimate,
        deps_type=MasteryDeps,
        retries=3,
        system_prompt=(
            "You are a teaching diagnostics calibrator.\n"
            "You will be given quiz attempts and summary stats.\n\n"
            "Return a structured MasteryEstimate:\n"
            "- mastery in [0,1]\n"
            "- calibration_delta in [-0.15, 0.15]\n"
            "- confidence in [0,1]\n"
            "- evidence: 1-3 short bullets (free text is OK)\n"
            "- next_action optional\n\n"
            "Important:\n"
            "If you mention counts, use the provided Correct and Total numbers.\n"
            "Do not invent a different number of attempts."
        ),
    )

    @agent.system_prompt
    async def add_context(ctx: RunContext[MasteryDeps]) -> str:
        d = ctx.deps
        attempt_lines = []
        for a in d.attempts:
            attempt_lines.append(
                f"- {a.uid} | diff={a.difficulty} | "
                f"user={a.user_answer} | correct={a.correct_answer} | "
                f"{'correct' if a.is_correct else 'wrong'} | time={a.response_time_sec:.1f}s"
            )
        return (
            f"Deck: {d.deck_id}\n"
            f"Correct={d.correct_count}, Total={d.total_count}, base_score={d.base_score:.2f}\n"
            f"avg_difficulty={d.avg_difficulty:.2f}, avg_time_sec={d.avg_time_sec:.1f}\n"
            f"Attempts:\n" + "\n".join(attempt_lines)
        )

    # 兼容不同 PydanticAI 版本：优先用 output_validator，没有就用 result_validator
    validator_decorator = getattr(agent, "output_validator", None)
    if validator_decorator is None:
        validator_decorator = getattr(agent, "result_validator")

    @validator_decorator
    def validate_mastery(ctx: RunContext[MasteryDeps], out: MasteryEstimate) -> MasteryEstimate:
        """
        Enforce logical consistency between mastery and (base_score + delta).
        If too inconsistent, ask the model to retry.
        """
        expected = ctx.deps.base_score + out.calibration_delta
        if abs(out.mastery - expected) > 0.10:
            raise ModelRetry(
                f"mastery should be close to base_score + calibration_delta. "
                f"base_score={ctx.deps.base_score:.2f}, delta={out.calibration_delta:+.2f}, "
                f"expected mastery≈{expected:.2f}. Please adjust mastery accordingly."
            )
        return out

    return agent



# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    print("[run_mastery_quiz] main() entered.", flush=True)

    questions = load_question_bank(QBANK_FILE)
    deck_id = questions[0].deck_id

    k = min(NUM_QUESTIONS, len(questions))
    sampled = random.sample(questions, k=k)

    print("=" * 80, flush=True)
    print(f"Loaded {len(questions)} questions. Randomly sampled {k}.", flush=True)
    print("=" * 80, flush=True)

    attempts: List[Attempt] = []

    for i, q in enumerate(sampled, start=1):
        print("\n" + "-" * 80, flush=True)
        print(f"Q{i}/{k} | Topic: {q.topic_id} | Difficulty: {q.difficulty}", flush=True)
        print(q.stem, flush=True)

        valid_keys = [x for x in ["A", "B", "C", "D"] if x in q.options]
        for key in valid_keys:
            print(f"  {key}. {q.options[key]}", flush=True)

        start = time.time()
        ans = prompt_user_answer(valid_keys)
        rt = time.time() - start

        if ans == "Q":
            print("Quit early.", flush=True)
            break

        is_correct = (ans == q.answer)
        attempts.append(
            Attempt(
                uid=q.uid,
                topic_id=q.topic_id,
                difficulty=q.difficulty,
                knowledge_tags=q.knowledge_tags,
                user_answer=ans,
                correct_answer=q.answer,
                is_correct=is_correct,
                response_time_sec=rt,
            )
        )

        print(f"Your answer: {ans} | Correct: {q.answer} | {'OK' if is_correct else 'WRONG'}", flush=True)

    if not attempts:
        print("No attempts recorded. Exit.", flush=True)
        return

    correct_count, total_count, base_score, avg_diff, avg_time = compute_stats(attempts)

    print("\n" + "=" * 80, flush=True)
    print(f"Finished: {correct_count}/{total_count} correct | base_score={base_score:.2f}", flush=True)
    print("=" * 80, flush=True)

    agent = build_mastery_agent()
    deps = MasteryDeps(
        deck_id=deck_id,
        attempts=attempts,
        correct_count=correct_count,
        total_count=total_count,
        base_score=base_score,
        avg_difficulty=avg_diff,
        avg_time_sec=avg_time,
    )

    print("\nCalling LLM for mastery estimate (structured output)...", flush=True)
    resp = agent.run_sync("Estimate mastery and provide evidence.", deps=deps)
    out: MasteryEstimate = resp.output

    fixed_evidence, was_corrected = sanitize_evidence(
        llm_lines=out.evidence,
        correct=correct_count,
        total=total_count,
        base=base_score,
    )
    out.evidence = fixed_evidence

    # If we had to correct hallucinated numbers, cap confidence a bit
    if was_corrected:
        out.confidence = min(out.confidence, 0.6)

    print("\nMasteryEstimate JSON (final):", flush=True)
    print(out.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[run_mastery_quiz] ERROR (stack trace):", flush=True)
        traceback.print_exc()
        raise
