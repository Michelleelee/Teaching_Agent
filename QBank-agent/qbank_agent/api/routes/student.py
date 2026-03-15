"""
Student endpoints for fetching and answering questions.
"""
from fastapi import APIRouter, HTTPException
from typing import List

from qbank_agent.api import schemas, services

router = APIRouter()

@router.get("/decks", response_model=schemas.DeckListResponse)
def get_decks():
    """List all available MCQ decks for students."""
    decks = services.list_available_decks()
    return schemas.DeckListResponse(decks=decks)

@router.get("/decks/{deck_id}/mcqs", response_model=schemas.DeckQuestionsResponse)
def get_deck_mcqs(deck_id: str):
    """
    Get all questions for a specific deck.
    Answers and rationales are stripped out to prevent cheating.
    """
    deck_data = services.get_deck_questions(deck_id)
    if not deck_data:
        raise HTTPException(status_code=404, detail=f"Deck '{deck_id}' not found.")
        
    return schemas.DeckQuestionsResponse(**deck_data)

@router.post("/evaluate", response_model=schemas.SubmitAnswerResponse)
def submit_answer(payload: schemas.SubmitAnswerRequest):
    """
    Submit a student's answer for evaluation.
    Returns whether it was correct, along with the rationale and evidence.
    """
    eval_result = services.evaluate_student_answer(
        deck_id=payload.deck_id,
        question_id=payload.question_id,
        selected_option=payload.selected_option
    )
    
    if not eval_result:
        raise HTTPException(status_code=404, detail="Question or deck not found, or evaluation failed.")
        
    return schemas.SubmitAnswerResponse(**eval_result)
