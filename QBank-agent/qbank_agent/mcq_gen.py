import os
import json
import time
import difflib
import questionary
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from . import config

# ----------------------------
# Master-level prompts (Evidence-grounded MCQ)
# ----------------------------
SYSTEM_PROMPT_MCQ = """You are an elite university professor and assessment designer in Pattern Recognition / Machine Learning.

Your mission is to write **challenging, high-diagnostic-value** multiple-choice questions (MCQs) using ONLY the provided slide content.

## STRICT EVIDENCE POLICY
- You may ONLY use the provided slide page content as evidence.
- Every question MUST include evidence_quotes copied VERBATIM from the slide content.
- Evidence must be sufficient to justify the correct answer and refute all distractors.
- Exactly ONE correct option (A/B/C/D).

## PEDAGOGICAL GOALS (CRITICAL)
1. **Cross-concept integration is the top priority.**
   - Whenever feasible, design questions that CONNECT two or more concepts from the topic.
   - Tag ALL relevant concepts in the knowledge_tags field.

2. Each question MUST achieve at least ONE of:
   - **Expose a common misconception** (e.g., confusing sigmoid vs softmax, MLE vs MAP)
   - **Require reasoning across multiple facts** from the slides (not just reading one sentence)
   - **Test boundary distinction** between two related concepts that students commonly conflate
   - **Demand application/interpretation** of a formula or principle, not just reciting it

3. **EXPLICITLY BANNED question types:**
   - Pure verbatim recall (e.g., "What does slide X say about Y?")
   - Trivial factual lookups
   - Questions answerable by keyword-matching a single sentence
   - "All/none of the above" options
   - Trick negations ("Which is NOT...") unless it genuinely probes understanding

4. **Distractor quality is paramount:**
   - Every distractor (wrong option) must be PLAUSIBLE — it should reflect a real misconception or a related but incorrect concept from the slides.
   - At least one distractor should be the answer to a DIFFERENT but related concept (classic confusion trap).

## DIFFICULTY SCALE (1-3)
- 1 (Basic Understanding): Tests clear definitions or direct relationships stated in a single slide. Requires recognizing correct terminology or basic facts without complex reasoning.
- 2 (Intermediate Connection & Application): Requires interpreting or connecting two pieces of information, or distinguishing between two commonly confused concepts. Cannot be answered by simply reading one sentence.
- 3 (Advanced Synthesis & Deep Reasoning): Requires multi-concept integration, applying principles, or navigating highly plausible misconception traps. Often draws evidence across multiple slide pages.

**Target distribution: 1-2 questions at difficulty 1; 2-3 questions at difficulty 2; 1-2 questions at difficulty 3.**

## MULTI-PAGE EVIDENCE
- Questions MAY (and are encouraged to) draw evidence from MULTIPLE slide pages within the same topic.
- When a question uses evidence from multiple pages, list ALL pages in the source.pages array and provide evidence_quotes from each.

OUTPUT MUST BE VALID JSON and follow the schema exactly."""

USER_PROMPT_TEMPLATE_MCQ = """
Generate {questions_per_topic} challenging, high-quality MCQs for the topic below.

Topic:
- deck_id: {deck_id}
- topic: {topic}
- topic_id: {topic_id}
- slide_range: {slide_range}

Concept blueprint (use for cross-concept question design; NOT as evidence):
{concepts_json}

Slide content (ONLY source of truth; quotes must match exactly):
{slides_content_json}

TASK:
- Generate exactly {questions_per_topic} MCQs.
- **Cross-concept integration**: design questions that connect >=2 concepts where possible. Tag all relevant concepts in knowledge_tags.
- **Difficulty distribution**: 1-2 questions at difficulty 1; 2-3 at difficulty 2; 1-2 at difficulty 3.
- **Multi-page evidence**: questions may draw from multiple slide pages. Use source.pages to list all referenced pages.
- **Distractor quality**: every wrong option must be plausible and reflect a real misconception.

Anti-patterns to AVOID:
- "What does slide X define as ...?" (pure recall)
- Options that are obviously absurd or unrelated to the topic
- Questions where the answer is a verbatim copy of one slide sentence

Output JSON schema (STRICT):
{{
  "topic_id": "{topic_id}",
  "topic": "{topic}",
  "questions": [
    {{
      "question_id": "unique_id",
      "topic_id": "{topic_id}",
      "knowledge_tags": ["tag1", "tag2"],
      "difficulty": 3,
      "stem": "Question text that requires reasoning, not just recall",
      "options": {{
        "A": "Plausible option reflecting a real concept",
        "B": "Plausible option reflecting a common misconception",
        "C": "Plausible option from a related but different concept",
        "D": "Plausible option with a subtle error"
      }},
      "answer": "A",
      "rationale": "Detailed explanation of WHY the answer is correct AND why each distractor is wrong, citing evidence.",
      "source": {{
        "page": 15,
        "pages": [15, 16]
      }},
      "evidence_quotes": [
        "...verbatim quote from slide page 15...",
        "...verbatim quote from slide page 16..."
      ]
    }}
  ]
}}

Hard constraints:
- evidence_quotes MUST be exact substrings of the provided slide content.
- Exactly one correct option.
- No external knowledge allowed.
- Each question should have a detailed rationale explaining the reasoning.
"""

def extract_first_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text

def call_openai_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "Missing OpenAI SDK. Please run: pip install openai\n"
        ) from e

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    last_err = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=config.OPENAI_MODEL,
                temperature=config.TEMPERATURE,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = getattr(resp, "output_text", None) or str(resp)
            return json.loads(extract_first_json_object(raw))
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"OpenAI call failed after {config.MAX_RETRIES} attempts: {last_err}")

def slide_to_lines(slide: Dict[str, Any], page_for_ids: int) -> List[Dict[str, str]]:
    lines_out = []
    title = (slide.get("title") or "").strip()
    if title:
        lines_out.append({"line_id": f"p{page_for_ids}_title", "text": title})

    content = slide.get("content") or ""
    raw = [ln.strip() for ln in (content or "").split("\n")]
    kept = []
    prev = None
    for ln in raw:
        if not ln:
            continue
        low = ln.lower()
        if "copyright" in low and "reserved" in low:
            continue
        if ln == prev:
            continue
        kept.append(ln)
        prev = ln
        
    for i, ln in enumerate(kept, start=1):
        lines_out.append({"line_id": f"p{page_for_ids}_l{i}", "text": ln})
    return lines_out

def collect_topic_slide_contents(deck: Dict[str, Any], topic_obj: Dict[str, Any]) -> Dict[str, Any]:
    page_to_slide = {}
    for slide in deck.get("slides", []):
        for p in slide.get("pages", [slide.get("page")]):
            page_to_slide[p] = slide

    pages = set()
    for c in topic_obj.get("concepts", []):
        for p in c.get("supporting_slides", []):
            pages.add(p)
    
    if "page_range" in topic_obj:
        pr = topic_obj["page_range"]
        if isinstance(pr, list) and len(pr) >= 2:
            pages.update(range(int(pr[0]), int(pr[1]) + 1))
        elif isinstance(pr, list) and len(pr) == 1:
            pages.add(int(pr[0]))

    slides_payload = []
    for p in sorted(pages):
        slide = page_to_slide.get(p)
        if not slide:
            continue

        content = (slide.get("content") or "").strip()
        if not content:
            continue

        slides_payload.append({
            "page": p,
            "title": slide.get("title", ""),
            "content": content
        })

    return { "slides": slides_payload }

def is_fuzzy_match(quote: str, content: str, threshold: float = 0.95) -> bool:
    q_clean = quote.replace("\n", " ").strip()
    c_clean = content.replace("\n", " ").strip()
    
    if not q_clean:
        return True
        
    matcher = difflib.SequenceMatcher(None, q_clean, c_clean)
    match_len = sum(m.size for m in matcher.get_matching_blocks())
    ratio = match_len / len(q_clean) if len(q_clean) > 0 else 0
    return ratio >= threshold

def validate_mcq_output(mcq_obj: Dict[str, Any], slides_content_json: Dict[str, Any]):
    page_to_content = {
        s["page"]: s.get("content", "")
        for s in slides_content_json.get("slides", [])
    }

    valid_questions = []
    errors = []

    for q in mcq_obj.get("questions", []):
        ok = True
        if q.get("answer") not in ["A", "B", "C", "D"]:
            ok = False
            errors.append(f"{q.get('question_id')}: invalid answer")

        source = q.get("source", {})
        pages = source.get("pages", [])
        if not pages:
            single_page = source.get("page")
            if single_page is not None:
                pages = [single_page]

        all_content = ""
        valid_pages = []
        for p in pages:
            c = page_to_content.get(p, "")
            if c:
                all_content += "\n" + c
                valid_pages.append(p)

        if not all_content.strip():
            ok = False
            errors.append(f"{q.get('question_id')}: invalid page source (pages={pages})")

        for quote in q.get("evidence_quotes", []):
            found = False
            contents_to_check = [page_to_content.get(p, "") for p in valid_pages] + [all_content]
            for content in contents_to_check:
                if not content:
                    continue
                if quote in content:
                    found = True
                    break
                if quote.replace("\n", " ") in content.replace("\n", " "):
                    found = True
                    break
                if is_fuzzy_match(quote, content, threshold=0.90):
                    found = True
                    break

            if not found:
                ok = False
                errors.append(f"{q.get('question_id')}: quote not found in pages {pages}")

        if ok:
            valid_questions.append(q)

    return {
        "topic_id": mcq_obj.get("topic_id"),
        "topic": mcq_obj.get("topic"),
        "questions": valid_questions
    }, errors

def generate_mcq_for_deck(deck: Dict[str, Any], skeleton: Dict[str, Any], save_path: Optional[Path] = None) -> Dict[str, Any]:
    deck_id = skeleton.get("deck_id", deck.get("source_file", "unknown"))

    output = {
        "deck_id": deck_id,
        "difficulty_scale": config.DIFFICULTY_SCALE,
        "questions_per_topic": config.QUESTIONS_PER_TOPIC,
        "topics": []
    }

    # Support both new syllabus format and old skeleton format
    topics = skeleton.get("syllabus", skeleton.get("topics", []))
    total_topics = len(topics)
    print(f"   📚 开始为文档提取题库，共计划处理 {total_topics} 个知识主题...")

    for t_idx, topic in enumerate(topics):
        topic_id = topic.get("topic_id", f"topic_{t_idx + 1}")
        topic_name = topic.get("topic", "")
        slide_range = topic.get("page_range", topic.get("slide_range", []))

        print(f"\n      ▶️  正在处理主题 [{t_idx + 1}/{total_topics}]: {topic_name}")

        slides_content_json = collect_topic_slide_contents(deck, topic)

        if not slides_content_json.get("slides"):
            print(f"         ⚠️ 警告：该主题未找到相关的幻灯片内容 (跳过)")
            output["topics"].append({
                "topic_id": topic_id,
                "topic": topic_name,
                "slide_range": slide_range,
                "questions": [],
                "warnings": ["No slide evidence collected for this topic."]
            })
            continue

        concepts = topic.get("sub_points", topic.get("concepts", []))
        concepts_json = json.dumps(concepts, ensure_ascii=False, indent=2)
        user_prompt = USER_PROMPT_TEMPLATE_MCQ.format(
            questions_per_topic=config.QUESTIONS_PER_TOPIC,
            deck_id=deck_id,
            topic=topic_name,
            topic_id=topic_id,
            slide_range=slide_range,
            concepts_json=concepts_json,
            slides_content_json=json.dumps(slides_content_json, ensure_ascii=False, indent=2)
        )
        
        last_errors = []
        mcq_obj = None

        for regen in range(config.REGEN_ATTEMPTS_PER_TOPIC + 1):
            print(f"         🤖 调用大模型生成 {config.QUESTIONS_PER_TOPIC} 道高质量考题... (第 {regen + 1} 次尝试)", end="", flush=True)
            try:
                mcq_obj = call_openai_json(SYSTEM_PROMPT_MCQ, user_prompt)
                print(" 完成!")
                
                print("         🔍 对生成的题目进行严格的证据链校验...", end="", flush=True)
                mcq_obj, errs = validate_mcq_output(mcq_obj, slides_content_json)
                last_errors = errs

                valid_count = len(mcq_obj.get("questions", []))
                print(f" 完成! (有效题目: {valid_count} 道)")

                if valid_count >= max(2, config.QUESTIONS_PER_TOPIC - 1):
                    print("         ✅ 题目质量与数量达标，进入下一环节")
                    break

                print(f"         ⚠️ 验证不充分 (剩余尝试: {config.REGEN_ATTEMPTS_PER_TOPIC - regen} 次), 将附带反馈重新生成...")
                feedback = "\n\nVALIDATION FAILED. Fix and regenerate.\n" + "\n".join(errs[:8])
                user_prompt = user_prompt + feedback
            except Exception as e:
                print(f"\n         ❌ 主题 {topic_id} 生成异常: {e}")

        output["topics"].append({
            "topic_id": topic_id,
            "topic": topic_name,
            "slide_range": slide_range,
            "questions": (mcq_obj.get("questions", []) if mcq_obj else []),
            "warnings": (last_errors if last_errors else [])
        })

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"   💾 当前文档的全部题库已保存至: {save_path if save_path else '未指定路径'}")
    return output

def find_matching_files() -> List[Tuple[Path, Path]]:
    pairs = []
    for deck_path in sorted(config.JSONS_DIR.glob("*.json")):
        stem = deck_path.stem
        # Use syllabus directory instead of old concept skeleton
        skel_path = config.SYLLABUS_DIR / f"{stem}_syllabus.json"
        if skel_path.exists():
            pairs.append((deck_path, skel_path))
    return pairs

def process_all_mcqs(headless: bool = False):
    config.setup_directories()
    pairs = find_matching_files()
    
    total_pairs = len(pairs)
    if total_pairs == 0:
        print("\n🧠 没有发现待处理的知识骨架文件。请先运行 PDF 解析和骨架提取。")
        return []

    print(f"\n🧠 开始批量生成客观考题 (MCQ)...")
    print(f"   AI 模型: {config.OPENAI_MODEL}")
    print(f"   共发现 {total_pairs} 个已解析的课程大纲文件\n")

    # Build task list with status
    task_items = []
    for deck_path, skel_path in pairs:
        stem = deck_path.stem
        out_path = config.GEN_DIR / f"{stem}.mcq_by_topic_{config.QUESTIONS_PER_TOPIC}.json"
        is_done = out_path.exists()
        task_items.append((deck_path, skel_path, out_path, is_done))

    if headless:
        # Auto-select all pending (not yet generated) items
        selected_tasks = [(d, s, o, done) for d, s, o, done in task_items if not done]
        if not selected_tasks:
            print("\n   ✅ 所有 MCQ 文件均已存在，无需重新生成。")
            return [o for _, _, o, _ in task_items]
        print(f"   [Headless] 自动选中 {len(selected_tasks)} 个待生成任务")
    else:
        # Interactive prompt
        choices = [
            questionary.Choice(
                title="⏭️  [全局] 跳过 MCQ 生成阶段，直接进入下一环节",
                value="SKIP_ALL"
            )
        ]
        for deck_path, skel_path, out_path, is_done in task_items:
            status_tag = "✅ 已存在" if is_done else "⏳ 待生成"
            choices.append(
                questionary.Choice(
                    title=f"[{status_tag}] {deck_path.name}",
                    value=(deck_path, skel_path, out_path, is_done),
                )
            )

        print("   👇 请通过 [空格] 勾选/取消勾选要执行的任务，[上下箭头] 移动，[回车] 确认：")
        selected_tasks = questionary.checkbox(
            "选择要生成 MCQ 的文件 (默认跳过已存在的文件):",
            choices=choices,
            style=questionary.Style([('selected', 'fg:green bold'), ('instruction', 'fg:gray')])
        ).ask()

        if not selected_tasks or "SKIP_ALL" in selected_tasks:
            print("\n   ⏭️  已选择跳过 MCQ 生成，退出当前阶段。")
            return []
        
    results = []

    for idx, (deck_path, skel_path, out_path, is_done) in enumerate(selected_tasks):
        print("="*60)
        print(f"🚀 [任务开始] 为文档生成题库: {deck_path.name} ({idx + 1}/{len(selected_tasks)})")
        print("="*60)
        
        stem = deck_path.stem
        
        # If the user explicitly checks a task that is already done, we assume they want to force regenerate it
        if out_path.exists() and not is_done:
            # this shouldn't normally happen unless externally modified, but keeping as safeguard
            print(f"   ⏭️  MCQ 题库已存在，跳过生成: {out_path.name}")
            results.append(out_path)
            print(f"🎯 [任务完成] {deck_path.name} (已跳过)\n")
            continue
        elif out_path.exists() and is_done:
            print(f"   ⚠️  检测到文件已存在，但用户强制重新生成: {out_path.name}")

        with open(deck_path, "r", encoding="utf-8") as f:
            deck = json.load(f)
        with open(skel_path, "r", encoding="utf-8") as f:
            skeleton = json.load(f)

        generate_mcq_for_deck(deck, skeleton, save_path=out_path)
        results.append(out_path)
        
        print(f"🎯 [任务完成] {deck_path.name} 处理完毕\n")
        
    print("="*60)
    print(f"🎉 批量 MCQ 处理完毕！本次共执行 {len(results)}/{len(selected_tasks)} 套题库。")
    print("="*60 + "\n")
    return results
