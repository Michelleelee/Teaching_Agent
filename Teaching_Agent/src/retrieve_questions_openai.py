import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI

from hybrid_retrieval import HybridRetriever


# -------- Config --------
CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"
QBANK_JSON = Path("Lecture 1 Foundations Bayes MLE and ERM.mcq_by_topic_8.json")

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5


def embed_query(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    if not QBANK_JSON.exists():
        raise FileNotFoundError(f"Missing question bank JSON: {QBANK_JSON.resolve()}")

    # Load Chroma
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(name=COLLECTION_NAME)
    retriever = HybridRetriever.from_qbank_json(QBANK_JSON, collection=col)

    # OpenAI client
    oai = OpenAI()

    # -------- Query text (replace with your slide/concept text) --------
    query_text = "Bayes rule, prior, likelihood, posterior, MAP vs MLE. Key idea and common confusion."
    print("Query:", query_text)
    print("-" * 80)

    q_emb = embed_query(oai, query_text)

    res = retriever.query(query_text=query_text, query_embedding=q_emb, n_results=TOP_K)

    for rank, (qid, doc, meta, score, dist, bm25_score) in enumerate(
        zip(
            res.ids,
            res.documents,
            res.metadatas,
            res.hybrid_scores,
            res.vector_distances,
            res.bm25_scores,
        ),
        start=1,
    ):
        dist_text = f"{dist:.4f}" if dist is not None else "n/a"
        print(f"[{rank}] id={qid}  hybrid={score:.4f}  vec_dist={dist_text}  bm25={bm25_score:.4f}  meta={meta}")
        print(doc[:500].strip())
        print("-" * 80)

    print("Done.")


if __name__ == "__main__":
    main()
