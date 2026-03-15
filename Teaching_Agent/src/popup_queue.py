# src/popup_queue.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def emit_popup(queue_path: str | Path, payload: dict[str, Any]) -> None:
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
