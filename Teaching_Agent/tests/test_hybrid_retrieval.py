from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from hybrid_retrieval import BM25Index, HybridEntry, HybridRetriever


class FakeCollection:
    def __init__(self, result):
        self.result = result

    def query(self, **kwargs):
        return self.result


def test_bm25_prefers_lexical_match():
    entries = [
        HybridEntry(uid="topic_a:q1", document="posterior prior bayes rule", metadata={"topic_id": "topic_a"}),
        HybridEntry(uid="topic_b:q2", document="support vector machine margin", metadata={"topic_id": "topic_b"}),
    ]

    index = BM25Index(entries)
    hits = index.query("posterior prior", top_k=2)

    assert len(hits) == 1
    assert hits[0][0] == "topic_a:q1"
    assert hits[0][1] > 0


def test_hybrid_retriever_merges_vector_and_bm25_and_applies_filter():
    entries = [
        HybridEntry(uid="topic_a:q1", document="posterior prior bayes rule", metadata={"topic_id": "topic_a", "question_id": "q1"}),
        HybridEntry(uid="topic_b:q2", document="support vector machine margin", metadata={"topic_id": "topic_b", "question_id": "q2"}),
        HybridEntry(uid="topic_a:q3", document="likelihood posterior evidence", metadata={"topic_id": "topic_a", "question_id": "q3"}),
    ]

    collection = FakeCollection(
        {
            "ids": [["topic_b:q2", "topic_a:q1", "stale:q9"]],
            "metadatas": [[
                {"topic_id": "topic_b", "question_id": "q2"},
                {"topic_id": "topic_a", "question_id": "q1"},
                {"topic_id": "stale", "question_id": "q9"},
            ]],
            "distances": [[0.05, 0.25, 0.1]],
        }
    )

    retriever = HybridRetriever(entries, collection=collection)
    result = retriever.query(query_text="posterior prior", query_embedding=[0.1, 0.2], n_results=3)

    assert result.ids[0] == "topic_a:q1"
    assert "topic_b:q2" in result.ids

    filtered = retriever.query(
        query_text="posterior prior",
        query_embedding=[0.1, 0.2],
        n_results=3,
        where_filter={"topic_id": "topic_a"},
    )

    assert filtered.ids
    assert all(meta["topic_id"] == "topic_a" for meta in filtered.metadatas)
    assert "topic_b:q2" not in filtered.ids
