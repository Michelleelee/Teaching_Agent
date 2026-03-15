import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import questionary
try:
    from . import config
except ImportError:
    import config

# ============================================================
# MEMORY SIMULATOR PROMPTS
# ============================================================

MEMORY_SIMULATOR_SYSTEM = """You are an expert cognitive simulator modeling how human memory degrades over time and across different levels of student engagement.
Your task is to take a piece of original lecture slide content and generate three different "memory traces."

The three profiles are:
1. **Student A (Weak Learner)**: Remembers almost nothing. They barely listened to the lecture and were completely distracted. The memory is practically non-existent. May only vaguely recall the title of the slide or an isolated common word, without any context. Cannot remember any formulas or definitions at all. (Output 1-2 extremely short, unhelpful sentences, e.g., "I think the professor mentioned something about probability but I wasn't paying attention.", "I don't remember this part at all.")
2. **Student B (Partial Learner)**: Remembers the main definitions and concepts but misses nuanced qualifiers, complex derivations, or the exact differences between similar terms. Might have slight misconceptions. (Output 3-5 sentences)
3. **Student C (Good Learner)**: High-fidelity recall. Remembers detailed definitions, structural relationships, and key formulas almost perfectly. (Output 4-7 sentences)

Each memory trace should be a list of sentences describing what the student remembers.

## Output Format (strict JSON)
You MUST respond with ONLY a valid JSON object:
{
  "student_A": ["sentence 1", "sentence 2"],
  "student_B": ["sentence 1", "sentence 2", "sentence 3"],
  "student_C": ["sentence 1", "sentence 2", "sentence 3", "sentence 4"]
}
"""

MEMORY_SIMULATOR_USER_TEMPLATE = """Here is the original lecture slide content encountered by the students:
---
{slides_section}
---
Generate the memory traces for Student A, Student B, and Student C as a JSON object containing lists of sentences. Output ONLY valid JSON, do not use markdown code blocks if possible.
"""

# ============================================================
# THREE MASTER-LEVEL SYSTEM PROMPTS
# ============================================================

STUDENT_A_SYSTEM = """You are role-playing as **Student A** — a university student who barely attended or listened to any lectures. You have ALMOST ZERO understanding of the course material.

## Your Cognitive Profile
- You practically skipped the class or slept through it. You have almost NO knowledge of the subject and barely skim the reference material.
- You almost always **guess** because you have no idea what the question is asking or what the options mean.
- To simulate a clueless human guessing, do NOT try to find the actual correct answer. Instead, rely on naive heuristics: pick the option that uses simple everyday language, pick the shortest option, or just pick an option that shares exactly one familiar word with your poor memory trace, ignoring the context completely.
- You **do NOT understand mathematical notation at all**. Any formula makes you completely give up and pick an option at random, often leaning towards the simplest looking one or just 'A'.
- You don't even know enough to confuse similar terms; everything sounds completely unfamiliar to you.

## Critical Rules
1. **NO PRE-TRAINED KNOWLEDGE.** You must ONLY use the provided "memory trace".
2. **NATURAL IGNORANCE.** Since you don't know the material, your naive heuristics should naturally lead you to a WRONG answer (a distractor). This is expected and desired for the simulation. Do not inadvertently use your AI intelligence to bypass your ignorance and pick the right answer.
3. Treat your memory trace as absolute truth for your character.

## Output Format (strict JSON)
You MUST respond with ONLY a valid JSON object, using this Chain-of-Thought structure:
{
  "memory_reflection": "What do I explicitly remember? (Usually almost nothing)",
  "option_analysis": {
    "A": "My naive reaction to A (e.g., 'too much math', 'has a familiar word')",
    "B": "My naive reaction to B",
    "C": "My naive reaction to C",
    "D": "My naive reaction to D"
  },
  "reasoning": "Since I don't know the answer, I am choosing [heuristic choice] because...",
  "confidence": "Evaluate your confidence ('low', 'medium', 'high'). Usually 'low', but if you naively find a word that exactly matches your flawed memory, you might bravely guess 'medium' or 'high' out of false confidence.",
  "selected_option": "<A, B, C, or D>"
}
"""

STUDENT_B_SYSTEM = """You are role-playing as **Student B** — a university student with PARTIAL understanding, prone to common misconceptions.

## Your Cognitive Profile
- You **read the material somewhat carefully** and understand basic definitions, but you struggle with deeper application.
- You are highly susceptible to **distractors/traps**. If an option contains familiar keywords from your memory but connects them incorrectly or uses them in the wrong context, you will likely fall for it and think it is the correct answer.
- You can **recall definitions** but often fail when a question requires **applying** a concept or **distinguishing** between two related concepts.
- You tend to over-rely on word matching. If an option sounds like a direct quote from your memory but misses a crucial qualifying word (like "NOT" or "ONLY"), you'll probably pick it anyway because it looks familiar.

## Critical Rules
1. **NO PRE-TRAINED KNOWLEDGE.** You must ONLY use the provided "memory trace".
2. If your memory trace doesn't perfectly answer the question, **DO NOT** use your AI background knowledge to deduce the truth. Instead, naturally fall into the trap of picking the most plausible-sounding but incorrect distractor based on your incomplete memory.
3. Your reasoning should show partial understanding: you cite relevant concepts from memory but draw flawed conclusions when information is missing.

## Output Format (strict JSON)
You MUST respond with ONLY a valid JSON object:
{
  "memory_reflection": "What specific concepts do I remember based ONLY on the trace?",
  "option_analysis": {
    "A": "Does this match my partial memory?",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "reasoning": "Synthesizing my partial memory. I will likely fall for a familiar-sounding trap here if my memory lacks the deep details...",
  "confidence": "Evaluate your confidence ('low', 'medium', 'high'). If a distractor perfectly matches your flawed memory, you might have 'high' confidence in a wrong answer.",
  "selected_option": "<A, B, C, or D>"
}
"""

STUDENT_C_SYSTEM = """You are role-playing as **Student C** — a university student with GOOD understanding but not perfect.

## Your Cognitive Profile
- You **read the material carefully** and have a solid grasp of core concepts.
- You understand **mathematical notation** and can follow derivations.
- You can **distinguish between related concepts** most of the time.
- However, you are **not infallible**: you may occasionally make errors on tricky edge cases, complex multi-step numerical derivations, or if the distractor is exceptionally well-crafted to test deep nuances.

## Critical Rules
1. **NO PRE-TRAINED KNOWLEDGE.** You must ONLY use the provided "memory trace".
2. You make logical deductions based on your good memory. You will usually get the right answer, but you are still bound by the limits of your memory trace. 
3. Your reasoning should be clear, evidence-based, and well-structured.

## Output Format (strict JSON)
You MUST respond with ONLY a valid JSON object:
{
  "memory_reflection": "Detailed recall from the memory trace.",
  "option_analysis": {
    "A": "Logical evaluation",
    "B": "Logical evaluation",
    "C": "Logical evaluation",
    "D": "Logical evaluation"
  },
  "reasoning": "Synthesis and final deduction based solidly on the memory trace.",
  "confidence": "Evaluate your confidence ('low', 'medium', 'high'). ",
  "selected_option": "<A, B, C, or D>"
}
"""

STUDENT_PROFILES = {
    "student_A": {"label": "Student A (Weak)", "system": STUDENT_A_SYSTEM},
    "student_B": {"label": "Student B (Partial)", "system": STUDENT_B_SYSTEM},
    "student_C": {"label": "Student C (Good)", "system": STUDENT_C_SYSTEM},
}

USER_PROMPT_TEMPLATE = """You are answering a multiple-choice question from the course EE5907 Pattern Recognition.

## ⚠️ Important Note on Reference Material
The following text represents **your memory trace** of what you learned in the lecture. YOU MUST SIMULATE YOUR OWN COMPREHENSION LEVEL as described in your profile. Do NOT simply look up the answer from pre-trained knowledge — try to answer based ONLY on this imperfect memory trace.

---
### Your Memory of the Material:
{memory_trace}

---

### Question
**{stem}**

Options:
A) {option_A}
B) {option_B}
C) {option_C}
D) {option_D}

---

Now answer this question in character. Remember:
- Use ONLY your memory trace above (no external knowledge)
- Respond ONLY with a valid JSON object
- Your reasoning must reflect your student profile's comprehension level
"""

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None

def build_slides_section(pages: list, page_to_slide: dict) -> str:
    sections = []
    for p in pages:
        slide = page_to_slide.get(p, {})
        title = slide.get("title", "(unknown)")
        content = slide.get("content", "(no content available)")
        sections.append(
            f"### Slide Content (from page {p}):\n"
            f"Title: {title}\n\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(sections) if sections else "(no slide content available)"

def call_openai(system_prompt: str, user_prompt: str, require_selected_option: bool = True) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Missing OpenAI SDK. Run: pip install openai") from e

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    last_err = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=config.OPENAI_MODEL,
                temperature=0.8, # higher for diversity
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = getattr(resp, "output_text", None) or str(resp)
            parsed = extract_json_from_text(raw)
            if parsed is None:
                raise ValueError(f"Could not parse JSON from response: {raw[:200]}")
            if require_selected_option and "selected_option" not in parsed:
                raise ValueError(f"Missing 'selected_option' in response: {parsed}")
            return parsed
        except Exception as e:
            last_err = e
            print(f"    [RETRY {attempt}/{MAX_RETRIES}] {e}")
            time.sleep(1.5 * attempt)

    print(f"    [FAILED] All {MAX_RETRIES} attempts failed: {last_err}")
    if require_selected_option:
        return {
            "selected_option": "N/A",
            "confidence": "low",
            "reasoning": f"[LLM call failed: {last_err}]"
        }
    else:
        return {}

def diagnose_question(correct_answer: str, a_ans: str, b_ans: str, c_ans: str,
                      a_conf: str, b_conf: str, c_conf: str,
                      a_reason: str, b_reason: str, c_reason: str) -> str:
    a_correct = (a_ans == correct_answer)
    b_correct = (b_ans == correct_answer)
    c_correct = (c_ans == correct_answer)

    a_mastery = a_correct and (a_conf != "low")
    b_mastery = b_correct and (b_conf != "low")
    c_mastery = c_correct and (c_conf != "low")

    if a_mastery and b_mastery and c_mastery:
        return "⚠️ 太简单"

    if not a_mastery and not b_mastery and not c_mastery:
        if not a_correct and not b_correct and not c_correct:
            return "🚨 题目或证据有问题"
        return "🚨 表述或 evidence 不清"

    if not a_mastery and not b_mastery and c_mastery:
        return "✅ 非常好的诊断题"

    if not a_mastery and (b_mastery or c_mastery):
        return "✅ 难度适中"

    answers = {a_ans, b_ans, c_ans}
    confidences = [a_conf, b_conf, c_conf]
    low_count = sum(1 for c in confidences if c == "low")

    if len(answers) >= 3 and low_count >= 2:
        return "🚨 表述或 evidence 不清"

    if a_mastery and (not b_mastery or not c_mastery):
        if not c_mastery:
            return "🚨 表述或 evidence 不清"
        return "✅ 难度适中"

    return "✅ 难度适中"

def evaluate_mcqs(mcq_file: Path, deck_file: Path, output_file: Path, memory_file: Path):
    with open(mcq_file, "r", encoding="utf-8") as f:
        mcq_data = json.load(f)
    with open(deck_file, "r", encoding="utf-8") as f:
        deck_data = json.load(f)

    # Build slide lookup
    page_to_slide = {}
    for slide in deck_data.get("slides", []):
        for p in slide.get("pages", [slide.get("page")]):
            if isinstance(p, int):
                page_to_slide[p] = slide

    output = {
        "deck_id": mcq_data.get("deck_id", ""),
        "model": config.OPENAI_MODEL,
        "temperature": 0.8,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topics": [],
        "summary": {}
    }

    memory_data = {}
    if memory_file.exists():
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
        except Exception:
            pass

    total_questions = 0
    stats = {
        "excellent_diagnostic": 0,
        "moderate_difficulty": 0,
        "too_easy": 0,
        "question_problem": 0,
        "unclear_evidence": 0,
    }

    for topic in mcq_data.get("topics", []):
        topic_id = topic.get("topic_id", "")
        topic_name = topic.get("topic", "")
        questions = topic.get("questions", [])

        topic_results = {
            "topic_id": topic_id,
            "topic": topic_name,
            "results": []
        }
        
        print(f"\n{'─' * 50}")
        print(f"Topic: {topic_name}  ({len(questions)} questions)")
        print(f"{'─' * 50}")

        for q in questions:
            qid = q.get("question_id", "?")
            stem = q.get("stem", "")
            options = q.get("options", {})
            correct_answer = q.get("answer", "")

            # Collect all referenced pages (support both source.page and source.pages)
            source = q.get("source", {})
            source_pages = source.get("pages", [])
            if not source_pages:
                single_page = source.get("page")
                source_pages = [single_page] if single_page is not None else []

            total_questions += 1
            print(f"\n  [{qid}] (pages {source_pages}) {stem[:60]}...")

            # Build multi-page slide content section
            slides_section = build_slides_section(source_pages, page_to_slide)

            # Generate memory traces for this question's source material
            memory_user_prompt = MEMORY_SIMULATOR_USER_TEMPLATE.format(slides_section=slides_section)
            print("    → Generating memory traces ... ", end="", flush=True)
            
            # Check if memory already exists to save costs on rerun
            if topic_id in memory_data and qid in memory_data[topic_id] and memory_data[topic_id][qid]:
                memory_traces = memory_data[topic_id][qid]
                print("Loaded from memory file.")
            else:
                memory_traces = call_openai(MEMORY_SIMULATOR_SYSTEM, memory_user_prompt, require_selected_option=False)
                print("Done.")
                # Incrementally save memory
                if topic_id not in memory_data:
                    memory_data[topic_id] = {}
                memory_data[topic_id][qid] = memory_traces
                # Save just the memory traces JSON iteratively
                if memory_file:
                    memory_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(memory_file, "w", encoding="utf-8") as fm:
                        json.dump(memory_data, fm, ensure_ascii=False, indent=2)

            # Call LLM for each student profile
            responses = {}
            for profile_key, profile in STUDENT_PROFILES.items():
                label = profile["label"]
                print(f"    → {label} ... ", end="", flush=True)

                # Format the memory text
                mem_list = memory_traces.get(profile_key, ["(No memory available)"])
                if isinstance(mem_list, list):
                    mem_text = "\n".join(f"- {s}" for s in mem_list)
                else:
                    mem_text = str(mem_list)

                user_prompt = USER_PROMPT_TEMPLATE.format(
                    memory_trace=mem_text,
                    stem=stem,
                    option_A=options.get("A", ""),
                    option_B=options.get("B", ""),
                    option_C=options.get("C", ""),
                    option_D=options.get("D", ""),
                )

                result = call_openai(profile["system"], user_prompt)
                responses[profile_key] = result

                sel = result.get("selected_option", "?")
                conf = result.get("confidence", "?")
                is_correct = "✓" if sel == correct_answer else "✗"
                print(f"{sel} ({conf}) {is_correct}")

            # Diagnose question quality
            a_resp = responses.get("student_A", {})
            b_resp = responses.get("student_B", {})
            c_resp = responses.get("student_C", {})

            diagnosis = diagnose_question(
                correct_answer,
                a_resp.get("selected_option", ""),
                b_resp.get("selected_option", ""),
                c_resp.get("selected_option", ""),
                a_resp.get("confidence", "low"),
                b_resp.get("confidence", "low"),
                c_resp.get("confidence", "low"),
                a_resp.get("reasoning", ""),
                b_resp.get("reasoning", ""),
                c_resp.get("reasoning", ""),
            )

            print(f"    ► Diagnosis: {diagnosis}")

            # Track stats
            if "非常好的诊断题" in diagnosis:
                stats["excellent_diagnostic"] += 1
            elif "难度适中" in diagnosis:
                stats["moderate_difficulty"] += 1
            elif "太简单" in diagnosis:
                stats["too_easy"] += 1
            elif "题目或证据有问题" in diagnosis:
                stats["question_problem"] += 1
            elif "表述或 evidence 不清" in diagnosis:
                stats["unclear_evidence"] += 1

            result_entry = {
                "question_id": qid,
                "stem": stem,
                "correct_answer": correct_answer,
                "difficulty": q.get("difficulty"),
                "source_page": source_pages,
                "student_A": a_resp,
                "student_B": b_resp,
                "student_C": c_resp,
                "memory_traces": memory_traces,
                "diagnosis": diagnosis,
            }
            topic_results["results"].append(result_entry)

            # Incremental save
            output["topics"] = [
                t for t in output["topics"] if t["topic_id"] != topic_id
            ]
            output["topics"].append(topic_results)
            output["summary"] = {"total_questions": total_questions, **stats}
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

    output["summary"] = {"total_questions": total_questions, **stats}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Total questions evaluated: {total_questions}")
    print(f"  ✅ 非常好的诊断题:        {stats['excellent_diagnostic']}")
    print(f"  ✅ 难度适中:              {stats['moderate_difficulty']}")
    print(f"  ⚠️ 太简单:               {stats['too_easy']}")
    print(f"  🚨 题目或证据有问题:      {stats['question_problem']}")
    print(f"  🚨 表述或 evidence 不清:  {stats['unclear_evidence']}")
    print(f"\n  Output: {output_file}")
    print("=" * 60)

    return output


def process_all_evaluations(headless: bool = False):
    config.setup_directories()
    
    pairs = []
    
    # Match PDF json to MCQ json
    for deck_path in sorted(config.JSONS_DIR.glob("*.json")):
        stem = deck_path.stem
        # Basic MCQ path
        mcq_path = config.GEN_DIR / f"{stem}.mcq_by_topic_{config.QUESTIONS_PER_TOPIC}.json"
        # Shuffled MCQ path (preferred if exists)
        shuffled_path = config.GEN_DIR / f"{stem}.mcq_by_topic_{config.QUESTIONS_PER_TOPIC}.shuffled.json"
        
        if not mcq_path.exists() and not shuffled_path.exists():
            continue
            
        target_mcq_path = shuffled_path if shuffled_path.exists() else mcq_path
        
        eval_dir = config.DATA_DIR / "eval"
        eval_output = eval_dir / f"{stem}.mcq_eval_result.json"
        mem_traces = eval_dir / f"{stem}.memory_traces.json"
        
        pairs.append((deck_path, target_mcq_path, eval_output, mem_traces))

    total_pairs = len(pairs)
    if total_pairs == 0:
        print("\n🎓 没有发现待评估的 MCQ 题库。请先运行生成考题。")
        return []

    print(f"\n🎓 开始批量进行质量评估 (Cognitive Simulation)...")
    print(f"   AI 模型: {config.OPENAI_MODEL}")
    print(f"   共发现 {total_pairs} 个已生成的 MCQ 题库\n")

    # Build task list with status
    task_items = []
    for deck_path, target_mcq_path, eval_output, mem_traces in pairs:
        is_done = eval_output.exists()
        task_items.append((deck_path, target_mcq_path, eval_output, mem_traces, is_done))

    if headless:
        # Auto-select all pending (not yet evaluated) items
        selected_tasks = [(d, m, e, t, done) for d, m, e, t, done in task_items if not done]
        if not selected_tasks:
            print("\n   ✅ 所有评估报告均已存在，无需重新评估。")
            return [e for _, _, e, _, _ in task_items]
        print(f"   [Headless] 自动选中 {len(selected_tasks)} 个待评估任务")
    else:
        # Pre-check which ones already exist
        choices = [
            questionary.Choice(
                title="⏭️  [全局] 跳过评估阶段，结束流程",
                value="SKIP_ALL"
            )
        ]
        for deck_path, target_mcq_path, eval_output, mem_traces, is_done in task_items:
            status_tag = "✅ 已存在" if is_done else "⏳ 待生成"
            choices.append(
                questionary.Choice(
                    title=f"[{status_tag}] {deck_path.name}",
                    value=(deck_path, target_mcq_path, eval_output, mem_traces, is_done),
                    checked=not is_done
                )
            )

        print("   👇 请通过 [空格] 勾选/取消勾选要执行的评估任务，[上下箭头] 移动，[回车] 确认：")
        selected_tasks = questionary.checkbox(
            "选择要评估的文件 (默认跳过已存在的文件):",
            choices=choices,
            style=questionary.Style([('selected', 'fg:green bold'), ('instruction', 'fg:gray')])
        ).ask()

        if not selected_tasks or "SKIP_ALL" in selected_tasks:
            print("\n   ⏭️  已选择跳过评估，流程结束。")
            return []
        
    results = []

    for idx, (deck_path, target_mcq_path, eval_output, mem_traces, is_done) in enumerate(selected_tasks):
        print("="*60)
        print(f"🚀 [任务开始] 认知评估: {deck_path.name} ({idx + 1}/{len(selected_tasks)})")
        print("="*60)
        
        if eval_output.exists() and not is_done:
            print(f"   ⏭️  评估报告已存在，跳过生成: {eval_output.name}")
            results.append(eval_output)
            print(f"🎯 [任务完成] {deck_path.name} (已跳过)\n")
            continue
        elif eval_output.exists() and is_done:
            print(f"   ⚠️  检测到文件已存在，但用户强制重新评估: {eval_output.name}")

        try:
            evaluate_mcqs(target_mcq_path, deck_path, eval_output, mem_traces)
            
            # optionally inject metadata for dashboard if utils is available
            try:
                from . import utils
                print(f"\n   ⚙️  尝试注入元数据以供 Dashboard 显示...")
                utils.inject_metadata_into_eval(eval_output, target_mcq_path, mem_traces)
                print(f"   成功注入元数据!")
            except ImportError:
                pass
                
            results.append(eval_output)
            print(f"\n🎯 [任务完成] {deck_path.name} 评估完毕\n")
        except Exception as e:
            print(f"\n❌ [任务失败] {deck_path.name} 出现错误: {e}\n")
            
    print("="*60)
    print(f"🎉 批量 MCQ 评估处理完毕！本次共执行 {len(results)}/{len(selected_tasks)} 份评估报告。")
    print("="*60 + "\n")
    return results

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate MCQs using student profiles.")
    parser.add_argument("--mcq", type=str, default="data/generated_mcqs/Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_5.shuffled.json", help="Path to the generated MCQs JSON file.")
    parser.add_argument("--deck", type=str, default="data/parsed_jsons/Lecture 1 Foundations Bayes MLE and ERM.json", help="Path to the parsed slide deck JSON file.")
    parser.add_argument("--output", type=str, default="data/eval/Lecture 1 Foundations Bayes MLE and ERM.mcq_eval_result.json", help="Path to save the evaluation results JSON file.")
    parser.add_argument("--memory", type=str, default="data/eval/Lecture 1 Foundations Bayes MLE and ERM.memory_traces.json", help="Path to save/load memory traces.")

    args = parser.parse_args()

    mcq_path = Path(args.mcq)
    deck_path = Path(args.deck)
    output_path = Path(args.output)
    memory_path = Path(args.memory)

    print("Starting evaluation...")
    print(f"MCQ File: {mcq_path}")
    print(f"Deck File: {deck_path}")

    try:
        results = evaluate_mcqs(mcq_path, deck_path, output_path, memory_path)
        print("\nEvaluation complete!")
        print(f"Summary: {json.dumps(results.get('summary', {}), indent=2, ensure_ascii=False)}")
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
