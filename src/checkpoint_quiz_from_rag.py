import json
import os
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv
import chromadb
from openai import OpenAI


CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"
QBANK_JSON = Path("Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic.json")

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5


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

    print("=" * 80)
    print(f"Selected: {uid} | topic_id={topic_id} | question_id={qid} | difficulty={diff}")
    print(stem)
    for k in ["A", "B", "C", "D"]:
        if k in options:
            print(f"  {k}. {options[k]}")
    print("=" * 80)


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env or environment.")
    if not QBANK_JSON.exists():
        raise FileNotFoundError(f"Missing question bank JSON: {QBANK_JSON.resolve()}")

    deck_id, qmap = load_qbank_map(QBANK_JSON)
    print(f"Loaded qbank deck_id={deck_id}, total_questions={len(qmap)}")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(name=COLLECTION_NAME)
    oai = OpenAI()

    # ---- 你要改的地方：把这里的 slide_text 换成“当前 slide/concept 的文本” ----
    slide_text = "Bayes rule and MAP vs MLE: how prior changes the objective; common confusion between P(A|B) and P(B|A)."
    print("\nSlide/Concept query text:")
    print(slide_text)
    print("-" * 80)

    q_emb = embed_query(oai, slide_text)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["metadatas", "distances"],
    )

    ids = res["ids"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print("Top-K candidates:")
    for i, (uid, meta, dist) in enumerate(zip(ids, metas, dists), start=1):
        print(f"[{i}] uid={uid} distance={dist:.4f} meta={meta}")
    print("-" * 80)

    # MVP：直接取 top1
    best_uid = ids[0]
    best_meta = metas[0]
    topic_id = best_meta.get("topic_id")
    qid = str(best_meta.get("question_id"))

    key = (topic_id, qid)
    if key not in qmap:
        raise RuntimeError(f"Retrieved candidate not found in JSON map: {key}")

    q = qmap[key]
    print_question(best_uid, topic_id, qid, q)

    valid = [k for k in ["A", "B", "C", "D"] if k in (q.get("options") or {})]
    ans = input(f"Your answer ({'/'.join(valid)}): ").strip().upper()
    correct = str(q.get("answer", "")).strip().upper()

    print(f"Your answer: {ans} | Correct: {correct} | {'OK' if ans == correct else 'WRONG'}")


if __name__ == "__main__":
    main()
