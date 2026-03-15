"""
Example entry point to demonstrate how to use the `qbank_agent` package.
"""

from qbank_agent import config, pdf_parser, syllabus_gen, mcq_gen, evaluation, dashboard, utils
import json

def run_pipeline():
    print("=== QBank-Agent Pipeline ===")
    
    # 1. Ensure folders exist
    config.setup_directories()
    print(f"Data directories initialized at: {config.DATA_DIR}")

    # 2. PDF Parsing (assuming you placed a PDF in data/input_slides/)
    # For demonstration, we'll run against all existing PDFs.
    print("\n--- 1. Parsing PDFs ---")
    parsed_files = pdf_parser.process_all_slides()
    if parsed_files:
        print(f"Parsed {len(parsed_files)} PDF(s) to JSON. Example: {parsed_files[0].name}")
    else:
        print("No PDFs found to parse in data/input_slides/")
        return

    # 3. Syllabus Generation
    print("\n--- 2. Generating Syllabus ---")
    syllabus_files = syllabus_gen.process_all_syllabus()
    if syllabus_files:
        print(f"Generated {len(syllabus_files)} syllabus file(s).")
    else:
        print("Syllabus generation skipped or failed.")

    # 4. MCQ Generation
    print("\n--- 3. Generating MCQs ---")
    mcq_files = mcq_gen.process_all_mcqs()
    if mcq_files:
        print(f"Generated MCQs for {len(mcq_files)} topics.")
        # 5. Shuffle Options for all generated MCQs
        print("\n--- 4. Shuffling Options ---")
        for mcq_file in mcq_files:
            utils.shuffle_mcq_file(mcq_file)
            print(f"Shuffled: {mcq_file.name}")
        
        # 6. Verify References for all
        print("\n--- 5. Verifying References ---")
        for mcq_file in mcq_files:
            shuffled_file = config.GEN_DIR / f"{mcq_file.stem}.shuffled.json"
            # Remove '.mcq_by_topic_X' part from the stem to get the original slide JSON name
            base_stem = mcq_file.name.split('.mcq_by_topic_')[0]
            slides_json = config.JSONS_DIR / f"{base_stem}.json"
            
            if slides_json.exists() and shuffled_file.exists():
                 verification_results = utils.verify_mcq_references(shuffled_file, slides_json)
                 print(f"[{base_stem}] Accuracy: {verification_results['accuracy_pct']}% - Matches: {verification_results['matched_quotes']}")
    else:
        print("MCQ generation skipped or no new MCQs generated.")

    # 7. Student Evaluation Simulation (Interactive Batch)
    print(f"\n--- 6. Evaluating Quality (Cognitive Simulation) ---")
    eval_files = evaluation.process_all_evaluations()
    
    if eval_files:
        print("\nPipeline execution complete! You can run the dashboard manually to view results.")
    else:
        print("\nEvaluation skipped or failed.")

if __name__ == "__main__":
    run_pipeline()
