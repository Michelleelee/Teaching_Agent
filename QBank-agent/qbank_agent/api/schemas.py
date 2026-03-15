"""
Pydantic schemas for the FastAPI backend.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Student Facing Schemas ---

class DeckOverview(BaseModel):
    deck_id: str
    filename: str
    difficulty_scale: Optional[str] = None
    questions_count: int

class DeckListResponse(BaseModel):
    decks: List[DeckOverview]

class Option(BaseModel):
    id: str  # A, B, C, D
    text: str

class StudentQuestion(BaseModel):
    question_id: str
    topic_id: str
    stem: str
    options: Dict[str, str] # e.g. {"A": "Text", "B": "Text"}
    difficulty: int
    knowledge_tags: List[str]

class DeckQuestionsResponse(BaseModel):
    deck_id: str
    questions: List[StudentQuestion]
    total: int

class SubmitAnswerRequest(BaseModel):
    deck_id: str
    question_id: str
    selected_option: str

class SubmitAnswerResponse(BaseModel):
    correct: bool
    correct_option: str
    rationale: str
    evidence_quotes: List[str]

# --- Admin Facing Schemas ---

class GenerateDeckRequest(BaseModel):
    deck_id: Optional[str] = None # If None, generate all

class GenerateDeckResponse(BaseModel):
    status: str
    message: str
    
class GenericMessageResponse(BaseModel):
    message: str
