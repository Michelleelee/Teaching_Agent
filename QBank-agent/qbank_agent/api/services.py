"""
Data Access Layer and utility functions for the API.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from qbank_agent import config

def _get_json_files(directory: Path, pattern: str = "*.json") -> List[Path]:
    if not directory.exists():
        return []
    return list(directory.glob(pattern))

def list_available_decks() -> List[Dict[str, Any]]:
    """List all available shuffled MCQ decks."""
    # Pattern is *.shuffled.json (or fallback to .mcq_eval_result.json if we want evaluated ones)
    # The eval_result is the final product containing diagnoses
    files = _get_json_files(config.GEN_DIR, "*.mcq_eval_result.json")
    if not files:
        # Fallback to shuffled if eval hasn't run
        files = _get_json_files(config.GEN_DIR, "*.shuffled.json")
        
    decks: List[Dict[str, Any]] = []
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Safely count questions
            q_count = 0
            for topic in data.get('topics', []):
                q_count += len(topic.get('questions', topic.get('results', [])))
                
            deck_id = data.get('deck_id', f.stem.replace(".mcq_eval_result", "").replace(".mcq_by_topic", "").replace(".shuffled", ""))
            
            decks.append({
                "deck_id": deck_id,
                "filename": f.name,
                "difficulty_scale": data.get("difficulty_scale"),
                "questions_count": q_count
            })
        except Exception as e:
            print(f"Error reading file {f}: {e}")
            continue
            
    return decks

def get_deck_questions(deck_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve all questions for a specific deck, omitting the answers and rationale.
    """
    files = _get_json_files(config.GEN_DIR, f"*{deck_id}*.json")
    
    # Prioritize evaluated files
    target_file = None
    for f in files:
        if "eval_result" in f.name:
            target_file = f
            break
    
    if not target_file:
        for f in files:
            if "shuffled" in f.name:
                target_file = f
                break
                
    if not target_file:
        return None

    try:
        with open(target_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        student_questions = []
        for topic in data.get('topics', []):
            topic_id = topic.get('topic_id', "Unknown")
            questions_list = topic.get('questions', topic.get('results', []))
            
            for q in questions_list:
                student_questions.append({
                    "question_id": q.get('question_id', "Unknown"),
                    "topic_id": topic_id,
                    "stem": q.get('stem', ""),
                    "options": q.get('options', {}), # Dict structure
                    "difficulty": q.get('difficulty', 1),
                    "knowledge_tags": q.get('knowledge_tags', [])
                })
        
        return {
            "deck_id": data.get('deck_id', deck_id),
            "questions": student_questions,
            "total": len(student_questions)
        }
    except Exception as e:
        print(f"Error parsing deck {deck_id}: {e}")
        return None

def evaluate_student_answer(deck_id: str, question_id: str, selected_option: str) -> Optional[Dict[str, Any]]:
    """Evaluate a student's answer against the backend JSON source of truth."""
    files = _get_json_files(config.GEN_DIR, f"*{deck_id}*.json")
    
    # Try eval result first, then shuffled
    for suffix in ["eval_result", "shuffled"]:
        target_file = next((f for f in files if suffix in f.name), None)
        if target_file:
            break
            
    if not target_file:
        return None
        
    try:
        with open(target_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        for topic in data.get('topics', []):
            questions_list = topic.get('questions', topic.get('results', []))
            for q in questions_list:
                if q.get('question_id') == question_id:
                    correct_ans = q.get('answer', q.get('correct_answer'))
                    is_correct = (selected_option.upper() == correct_ans.upper()) if correct_ans else False
                    
                    return {
                        "correct": is_correct,
                        "correct_option": correct_ans,
                        "rationale": q.get('rationale', 'Reasoning not provided.'),
                        "evidence_quotes": q.get('evidence_quotes', [])
                    }
        return None
        
    except Exception as e:
        print(f"Error evaluating answer for {deck_id}: {e}")
        return None
