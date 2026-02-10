import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from tqdm import tqdm


# -------- Config --------
QBANK_JSON = Path("Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_8.json")
CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"

EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
BATCH_SIZE = 64

MAX_RATIONALE_CHARS = 1500
MAX_STEM_CHARS = 1500


def build_embedding_text(topic_id: str, q: Dict[str, Any]) -> str:
    stem = (q.get("stem") or "").strip()[:MAX_STEM_CHARS]
    options = q.get("options") or {}
    rationale = (q.get("rationale") or "").strip()[:MAX_RATIONALE_CHARS]
    tags = q.get("knowledge_tags") or []
    diff = q.get("difficulty", 2)

    opt_lines = []
    for k in ["A", "B", "C", "D"]:
        if k in options and options[k] is not None:
            opt_lines.append(f"{k}) {str(options[k]).strip()}")
    opt_block = "\n".join(opt_lines)

    # Do NOT include correct answer letter in embedding text.
    text = (
        f"Topic: {topic_id}\n"
        f"Tags: {tags}\n"
        f"Difficulty: {diff}\n\n"
        f"Question: {stem}\n"
        f"Options:\n{opt_block}\n\n"
        f"Explanation: {rationale}"
    ).strip()

    if not text:
        text = f"Topic: {topic_id}"
    return text


def flatten_questions(data: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[Dict[str, Any]]]:
    deck_id = data.get("deck_id", "unknown_deck")
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for topic in data.get("topics", []):
        topic_id = topic.get("topic_id", "unknown_topic")
        for q in topic.get("questions", []):
            qid = str(q.get("question_id", "unknown_q"))
            uid = f"{topic_id}:{qid}"

            doc = build_embedding_text(topic_id, q)

            # ---- 핵심修改点：metadata里明确写入 topic_id + question_id，用于在线回表 ----
            meta = {
                "deck_id": deck_id,
                "topic_id": topic_id,
                "question_id": qid,
                "difficulty": int(q.get("difficulty", 2)),
                "answer": str(q.get("answer", "")).strip().upper(),
            }

            ids.append(uid)
            docs.append(doc)
            metas.append(meta)

    return deck_id, ids, docs, metas


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or export it.")

    if not QBANK_JSON.exists():
        raise FileNotFoundError(f"Question bank JSON not found: {QBANK_JSON.resolve()}")

    data = json.loads(QBANK_JSON.read_text(encoding="utf-8"))
    deck_id, ids, docs, metas = flatten_questions(data)

    print(f"Loaded deck_id={deck_id}, questions={len(ids)}")
    print(f"Chroma dir: {CHROMA_DIR.resolve()}")
    print(f"Embedding model: {EMBED_MODEL}")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    oai = OpenAI()

    for start in tqdm(range(0, len(ids), BATCH_SIZE), desc="Embedding & Upserting"):
        end = min(start + BATCH_SIZE, len(ids))
        batch_ids = ids[start:end]
        batch_docs = docs[start:end]
        batch_metas = metas[start:end]

        batch_emb = embed_batch(oai, batch_docs)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_emb,
        )

    print(f"Done. Collection='{COLLECTION_NAME}' stored at {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()
