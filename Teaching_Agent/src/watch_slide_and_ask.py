import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

from hybrid_retrieval import HybridRetriever


CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"
QBANK_JSON = Path("Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_8.json")
SLIDE_FILE = Path("current_slide.txt")

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5
ASK_N = 1  # 每次切页问几题：1 最像课堂 checkpoint


def embed_query(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def load_qbank_map(path: Path) -> Tuple[str, Dict[Tuple[str, str], dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    deck_id = data.get("deck_id", "unknown_deck")
    qmap: Dict[Tuple[str, str], dict] = {}

    for topic in data.get("topics", []):
        topic_id = topic.get("topic_id", "unknown_topic")
        for q in topic.get("questions", []):
            qid = str(q.get("question_id", "unknown_q"))
            qmap[(topic_id, qid)] = q
    return deck_id, qmap


def print_question(uid: str, topic_id: str, qid: str, q: dict) -> None:
    stem = (q.get("stem") or "").strip()
    options = q.get("options") or {}
    diff = q.get("difficulty", "")

    print("\n" + "=" * 80, flush=True)
    print(f"Slide changed -> ask question: {uid} | topic_id={topic_id} | question_id={qid} | difficulty={diff}", flush=True)
    print(stem, flush=True)
    for k in ["A", "B", "C", "D"]:
        if k in options:
            print(f"  {k}. {options[k]}", flush=True)
    print("=" * 80, flush=True)


def read_slide_text() -> str:
    if not SLIDE_FILE.exists():
        return ""
    return SLIDE_FILE.read_text(encoding="utf-8").strip()


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env or environment.")
    if not QBANK_JSON.exists():
        raise FileNotFoundError(f"Missing question bank JSON: {QBANK_JSON.resolve()}")

    deck_id, qmap = load_qbank_map(QBANK_JSON)
    print(f"Loaded qbank deck_id={deck_id}, total_questions={len(qmap)}", flush=True)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(name=COLLECTION_NAME)
    retriever = HybridRetriever.from_qbank_json(QBANK_JSON, collection=col)
    oai = OpenAI()

    last_slide = None
    asked_uids = set()  # 避免重复问同一题（简单去重）

    print(f"Watching {SLIDE_FILE.resolve()} ... (edit this file to simulate slide changes)\n", flush=True)

    while True:
        slide_text = read_slide_text()
        if slide_text and slide_text != last_slide:
            last_slide = slide_text
            print(f"\n[Slide Update] {slide_text}", flush=True)

            q_emb = embed_query(oai, slide_text)
            res = retriever.query(query_text=slide_text, query_embedding=q_emb, n_results=TOP_K)

            asked = 0
            for uid, meta in zip(res.ids, res.metadatas):
                if uid in asked_uids:
                    continue

                topic_id = meta.get("topic_id")
                qid = str(meta.get("question_id"))
                key = (topic_id, qid)
                if key not in qmap:
                    continue

                q = qmap[key]
                print_question(uid, topic_id, qid, q)

                valid = [k for k in ["A", "B", "C", "D"] if k in (q.get("options") or {})]
                ans = input(f"Your answer ({'/'.join(valid)}): ").strip().upper()
                correct = str(q.get("answer", "")).strip().upper()
                print(f"Your answer: {ans} | Correct: {correct} | {'OK' if ans == correct else 'WRONG'}", flush=True)

                asked_uids.add(uid)
                asked += 1
                if asked >= ASK_N:
                    break

        time.sleep(1)  # polling interval

if __name__ == "__main__":
    main()
