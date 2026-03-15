import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

OPTION_KEYS = ["A", "B", "C", "D"]

def shuffle_question_options(question: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    options = question.get("options", {})
    correct_key = question.get("answer", "")

    pairs = [(k, options[k]) for k in OPTION_KEYS if k in options]
    if not pairs:
        return question

    correct_text = options.get(correct_key, "")
    texts = [t for _, t in pairs]
    rng.shuffle(texts)

    new_options = {}
    new_answer = correct_key
    for i, key in enumerate(OPTION_KEYS[:len(texts)]):
        new_options[key] = texts[i]
        if texts[i] == correct_text:
            new_answer = key

    q_copy = dict(question)
    q_copy["options"] = new_options
    q_copy["answer"] = new_answer
    return q_copy

def shuffle_mcq_file(src_path: Path, seed: int | None = None) -> Path:
    rng = random.Random(seed)

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for topic in data.get("topics", []):
        topic["questions"] = [
            shuffle_question_options(q, rng)
            for q in topic.get("questions", [])
        ]

    out_name = src_path.stem + ".shuffled.json"
    out_path = src_path.parent / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path

# ----------------------------
# References Verification Tool
# ----------------------------

def normalize_text(text):
    if not text:
        return ""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def verify_mcq_references(mcq_path: Path, slides_path: Path) -> Dict[str, Any]:
    with open(mcq_path, 'r', encoding='utf-8') as f:
        mcq_data = json.load(f)
    with open(slides_path, 'r', encoding='utf-8') as f:
        slides_data = json.load(f)

    page_content_map = {}
    if 'slides' in slides_data:
        for slide in slides_data['slides']:
            content = normalize_text(slide.get('content', ''))
            pages = slide.get('pages', [])
            for page_num in pages:
                page_content_map[page_num] = {
                    'title': slide.get('title', 'Unknown'),
                    'content': content
                }

    total_questions = 0
    total_quotes = 0
    matched_quotes = 0
    missing_quotes = 0
    missing_details = []

    topics = mcq_data.get('topics', [])
    for topic in topics:
        questions = topic.get('questions', [])
        for q in questions:
            total_questions += 1
            qid = q.get('question_id', 'unknown')
            source_page = q.get('source', {}).get('page')
            evidence_quotes = q.get('evidence_quotes', [])
            
            if not evidence_quotes:
                continue

            if source_page is None:
                continue
            
            slide_info = page_content_map.get(source_page)
            if not slide_info:
                for quote in evidence_quotes:
                    total_quotes += 1
                    missing_quotes += 1
                    missing_details.append({
                        'qid': qid,
                        'page': source_page,
                        'quote': quote,
                        'reason': 'Page not found'
                    })
                continue

            slide_content = slide_info['content']
            for quote in evidence_quotes:
                total_quotes += 1
                norm_quote = normalize_text(quote)
                
                if norm_quote in slide_content:
                    matched_quotes += 1
                elif norm_quote.lower() in slide_content.lower():
                     matched_quotes += 1
                else:
                    missing_quotes += 1
                    missing_details.append({
                        'qid': qid,
                        'page': source_page,
                        'quote': quote,
                        'reason': 'Quote not found in text'
                    })

    accuracy_pct = (matched_quotes / total_quotes * 100) if total_quotes > 0 else 0
    return {
        "accuracy_pct": accuracy_pct,
        "matched_quotes": matched_quotes,
        "failed_matches": missing_quotes,
        "missing_details": missing_details
    }

# ----------------------------
# Injection Utility
# ----------------------------

def inject_metadata_into_eval(eval_file_path: Path, shuffled_file_path: Path, mem_file_path: Path):
    if not eval_file_path.exists():
        return

    with open(eval_file_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Inject options matching QID
    if shuffled_file_path.exists():
        with open(shuffled_file_path, "r", encoding="utf-8") as f:
            shuffled_data = json.load(f)
            
        options_lookup = {}
        for topic in shuffled_data.get("topics", []):
            for q in topic.get("questions", []):
                qid = q.get("question_id")
                if qid:
                    options_lookup[qid] = q.get("options", {})

        for topic in eval_data.get("topics", []):
            for result in topic.get("results", []):
                qid = result.get("question_id")
                if qid in options_lookup:
                    result["options"] = options_lookup[qid]

    # Inject memory matching QID
    if mem_file_path.exists():
        with open(mem_file_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)

        for topic in eval_data.get("topics", []):
            topic_id = topic.get("topic_id")
            for result in topic.get("results", []):
                qid = result.get("question_id")
                if topic_id in mem_data and qid in mem_data[topic_id]:
                    result["memory_traces"] = mem_data[topic_id][qid]

    with open(eval_file_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
