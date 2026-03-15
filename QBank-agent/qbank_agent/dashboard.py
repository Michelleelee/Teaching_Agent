import http.server
import socketserver
import webbrowser
import threading
import sys
from pathlib import Path
from urllib.parse import quote

# Add project root to sys.path to allow running standalone
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qbank_agent import config

# We serve from the PROJECT_ROOT
DIRECTORY = config.PROJECT_ROOT

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def start_server_in_thread(port: int = 8080):
    """Starts the Python HTTP server in a daemon thread."""
    httpd = socketserver.TCPServer(("", port), Handler)
    print(f"Serving {DIRECTORY} at http://localhost:{port}")
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return httpd

def launch_dashboard(eval_json_rel_path: str, pdf_rel_path: str, port: int = 8080):
    """
    Launch the web dashboard pointing to the specified json and pdf paths.
    Paths should be relative to PROJECT_ROOT (e.g., "data/generated_mcqs/...")
    """
    # Start server
    start_server_in_thread(port)
    
    # Construct URL with query parameters for auto-loading
    encoded_json = quote(eval_json_rel_path, safe='/')
    encoded_pdf = quote(pdf_rel_path, safe='/')
    
    url = f"http://localhost:{port}/mcq_eval_dashboard.html?json={encoded_json}&pdf={encoded_pdf}"
    
    print(f"Opening browser to {url}")
    webbrowser.open(url)
    print("Dashboard server is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch the MCQ Evaluation Dashboard")
    parser.add_argument(
        "--json", 
        type=str, 
        default="data/generated_mcqs/Lecture 1 Foundations Bayes MLE and ERM.mcq_eval_result.json", 
        help="Relative path to the eval result JSON"
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="data/input_slides/Lecture 1 Foundations Bayes MLE and ERM.pdf", 
        help="Relative path to the source slide deck PDF"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the dashboard server on"
    )
    
    args = parser.parse_args()
    
    launch_dashboard(
        eval_json_rel_path=args.json,
        pdf_rel_path=args.pdf,
        port=args.port
    )
