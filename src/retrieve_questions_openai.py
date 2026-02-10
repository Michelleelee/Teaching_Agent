import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI


# -------- Config --------
CHROMA_DIR = Path("chroma_qbank_openai")
COLLECTION_NAME = "weijia_qbank"

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

    # Load Chroma
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = chroma.get_collection(name=COLLECTION_NAME)

    # OpenAI client
    oai = OpenAI()

    # -------- Query text (replace with your slide/concept text) --------
    query_text = "Bayes rule, prior, likelihood, posterior, MAP vs MLE. Key idea and common confusion."
    print("Query:", query_text)
    print("-" * 80)

    q_emb = embed_query(oai, query_text)

    # Query Chroma
    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for rank, (qid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        print(f"[{rank}] id={qid}  distance={dist:.4f}  meta={meta}")
        print(doc[:500].strip())
        print("-" * 80)

    print("Done.")


if __name__ == "__main__":
    main()
