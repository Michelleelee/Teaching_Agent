"""
Admin endpoints for triggering backend pipelines.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional

from qbank_agent.api import schemas
from qbank_agent import pdf_parser, syllabus_gen, mcq_gen, evaluation, utils

router = APIRouter()

def _run_full_generation_pipeline(deck_id: Optional[str] = None):
    """
    Background task to run the generation pipeline.
    If deck_id is provided, it could try to filter, but right now
    the modules mostly process all files in the input directory.
    We just run the standard generation for now.
    """
    print(f"--- Triggered Background Generation Pipeline ---")
    try:
        # We can implement a more focused generation in the future.
        # Currently, triggering re-runs all necessary steps.
        pdf_parser.process_all_slides()
        syllabus_gen.process_all_syllabus()
        mcq_files = mcq_gen.process_all_mcqs()
        
        for f in mcq_files:
            if deck_id and deck_id not in f.name:
                continue # Skip if specific deck requested
                
            shuffled_file = utils.shuffle_mcq_file(f)
            # Evaluate
            from qbank_agent import config
            slides_json = config.JSONS_DIR / f"{f.stem.split('.')[0]}.json"
            if slides_json.exists():
                eval_output = config.GEN_DIR / f"{shuffled_file.stem.split('.')[0]}.mcq_eval_result.json"
                mem_traces = config.GEN_DIR / f"{shuffled_file.stem.split('.')[0]}.memory_traces.json"
                
                evaluation.evaluate_mcqs(shuffled_file, slides_json, eval_output, mem_traces)
                utils.inject_metadata_into_eval(eval_output, shuffled_file, mem_traces)
                
        print("--- Background Generation Pipeline Complete ---")
    except Exception as e:
        print(f"--- Background Generation Failed: {e} ---")

@router.post("/generate", response_model=schemas.GenerateDeckResponse)
def trigger_generation(payload: schemas.GenerateDeckRequest, background_tasks: BackgroundTasks):
    """
    Trigger the backend pipeline to generate MCQs from PDFs.
    This runs asynchronously in the background.
    """
    background_tasks.add_task(_run_full_generation_pipeline, payload.deck_id)
    
    msg = "Generation pipeline started in background."
    if payload.deck_id:
        msg = f"Generation pipeline started for deck '{payload.deck_id}' in background."
        
    return schemas.GenerateDeckResponse(status="started", message=msg)
