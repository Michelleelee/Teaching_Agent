import os
import sys
from pathlib import Path

# Ensure the parent directory is in the PYTHONPATH
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from qbank_agent.api.main import app

if __name__ == "__main__":
    import uvicorn
    # Default port for FastAPI is 8000
    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=True)
