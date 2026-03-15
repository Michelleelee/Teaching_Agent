from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


MAX_RATIONALE_CHARS = 1500
MAX_STEM_CHARS = 1500
DEFAULT_VECTOR_WEIGHT = 0.65
DEFAULT_BM25_WEIGHT = 0.35
DEFAULT_CANDIDATE_MULTIPLIER = 4
DEFAULT_MIN_CANDIDATES = 20

_TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(frozen=True)
class HybridEntry:
    uid: str
    document: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class HybridQueryResult:
    ids: List[str]
    metadatas: List[Dict[str, Any]]
    documents: List[str]
    hybrid_scores: List[float]
    vector_scores: List[float]
    bm25_scores: List[float]
    vector_distances: List[Optional[float]]


def tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall((text or "").lower())


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metadata_matches(metadata: Dict[str, Any], where_filter: Optional[Dict[str, Any]]) -> bool:
    if not where_filter:
        return True
    for key, expected in where_filter.items():
        if metadata.get(key) != expected:
            return False
    return True


def _normalize_bm25(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    if max_score <= 0:
        return {uid: 0.0 for uid in score_map}
    return {uid: score / max_score for uid, score in score_map.items()}


def _distance_to_similarity(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    if distance < 0:
        distance = 0.0
    return 1.0 / (1.0 + distance)


def build_qbank_search_text(topic_id: str, question: Dict[str, Any]) -> str:
    stem = _to_text(question.get("stem"))[:MAX_STEM_CHARS]
    options = question.get("options") or {}
    rationale = _to_text(question.get("rationale") or question.get("explanation"))[:MAX_RATIONALE_CHARS]

    raw_tags = question.get("knowledge_tags") or []
    if isinstance(raw_tags, str):
        tags = raw_tags.strip()
    else:
        tags = ", ".join(_to_text(tag) for tag in raw_tags if _to_text(tag))

    diff = _to_int(question.get("difficulty"), 2)

    option_lines = []
    for key in ["A", "B", "C", "D"]:
        value = options.get(key)
        if value is not None:
            option_lines.append(f"{key}) {_to_text(value)}")
    option_block = "\n".join(option_lines)

    text = (
        f"Topic: {topic_id}\n"
        f"Tags: {tags}\n"
        f"Difficulty: {diff}\n\n"
        f"Question: {stem}\n"
        f"Options:\n{option_block}\n\n"
        f"Explanation: {rationale}"
    ).strip()

    if not text:
        text = f"Topic: {topic_id}"
    return text


def load_qbank_entries(path: Path) -> List[HybridEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    deck_id = _to_text(data.get("deck_id")) or "unknown_deck"
    entries: List[HybridEntry] = []

    for topic in data.get("topics", []) or []:
        parent_topic_id = _to_text(topic.get("topic_id")) or "unknown_topic"
        for question in topic.get("questions", []) or []:
            if question.get("status") == "deprecated":
                continue

            topic_id = _to_text(question.get("topic_id")) or parent_topic_id
            qid = _to_text(question.get("question_id")) or "unknown_q"
            uid = f"{topic_id}:{qid}"
            entries.append(
                HybridEntry(
                    uid=uid,
                    document=build_qbank_search_text(topic_id, question),
                    metadata={
                        "deck_id": deck_id,
                        "topic_id": topic_id,
                        "question_id": qid,
                        "difficulty": _to_int(question.get("difficulty"), 2),
                        "answer": _to_text(question.get("answer")).upper(),
                    },
                )
            )

    return entries


def load_collection_entries(collection: Any) -> List[HybridEntry]:
    try:
        payload = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []

    entries: List[HybridEntry] = []
    for index, uid in enumerate(ids):
        document = documents[index] if index < len(documents) and documents[index] is not None else ""
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] is not None else {}
        entries.append(HybridEntry(uid=str(uid), document=_to_text(document), metadata=dict(metadata)))
    return entries


class BM25Index:
    def __init__(self, entries: Sequence[HybridEntry], *, k1: float = 1.5, b: float = 0.75):
        self.entries = list(entries)
        self.k1 = k1
        self.b = b
        self.doc_terms: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_length = 0.0
        self._build()

    def _build(self) -> None:
        doc_freqs: Counter[str] = Counter()
        total_length = 0

        for entry in self.entries:
            tokens = tokenize(entry.document)
            term_freqs = Counter(tokens)
            self.doc_terms.append(term_freqs)
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            total_length += doc_length
            for term in term_freqs:
                doc_freqs[term] += 1

        if self.entries:
            self.avg_doc_length = total_length / len(self.entries)

        total_docs = len(self.entries)
        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

    def query(
        self,
        query_text: str,
        *,
        allowed_uids: Optional[Iterable[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[tuple[str, float]]:
        query_terms = Counter(tokenize(query_text))
        if not query_terms:
            return []

        allowed = set(allowed_uids) if allowed_uids is not None else None
        results: List[tuple[str, float]] = []
        avg_doc_length = self.avg_doc_length or 1.0

        for index, entry in enumerate(self.entries):
            if allowed is not None and entry.uid not in allowed:
                continue

            doc_length = self.doc_lengths[index] or 1
            term_freqs = self.doc_terms[index]
            score = 0.0

            for term, qf in query_terms.items():
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue

                idf = self.idf.get(term)
                if idf is None:
                    continue

                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_length / avg_doc_length)
                score += qf * idf * ((tf * (self.k1 + 1.0)) / denom)

            if score > 0:
                results.append((entry.uid, score))

        results.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            return results[:top_k]
        return results


class HybridRetriever:
    def __init__(
        self,
        entries: Sequence[HybridEntry],
        *,
        collection: Any | None = None,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
        min_candidates: int = DEFAULT_MIN_CANDIDATES,
    ):
        self.collection = collection
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.candidate_multiplier = candidate_multiplier
        self.min_candidates = min_candidates
        self._set_entries(entries)

    @classmethod
    def from_qbank_json(cls, qbank_path: Path, *, collection: Any | None = None, **kwargs: Any) -> "HybridRetriever":
        return cls(load_qbank_entries(qbank_path), collection=collection, **kwargs)

    @classmethod
    def from_collection_snapshot(cls, collection: Any, **kwargs: Any) -> "HybridRetriever":
        return cls(load_collection_entries(collection), collection=collection, **kwargs)

    def _set_entries(self, entries: Sequence[HybridEntry]) -> None:
        self.entries = list(entries)
        self.uid_to_entry = {entry.uid: entry for entry in self.entries}
        self.bm25 = BM25Index(self.entries)

    def refresh_from_collection_snapshot(self) -> None:
        if self.collection is None:
            return
        self._set_entries(load_collection_entries(self.collection))

    def add_entry(self, uid: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        next_entries = self.entries + [
            HybridEntry(uid=str(uid), document=_to_text(document), metadata=dict(metadata or {}))
        ]
        self._set_entries(next_entries)

    def query(
        self,
        *,
        query_text: str,
        query_embedding: Optional[List[float]],
        n_results: int,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> HybridQueryResult:
        candidate_pool = max(n_results * self.candidate_multiplier, self.min_candidates)

        allowed_uids = [
            entry.uid for entry in self.entries
            if _metadata_matches(entry.metadata, where_filter)
        ]

        bm25_hits = self.bm25.query(query_text, allowed_uids=allowed_uids, top_k=candidate_pool)
        bm25_raw = {uid: score for uid, score in bm25_hits}
        bm25_scores = _normalize_bm25(bm25_raw)

        vector_raw, vector_meta = self._query_vector(
            query_embedding=query_embedding,
            n_results=candidate_pool,
            where_filter=where_filter,
        )

        candidate_ids: List[str] = []
        for uid in vector_raw:
            if uid not in candidate_ids:
                candidate_ids.append(uid)
        for uid, _ in bm25_hits:
            if uid not in candidate_ids:
                candidate_ids.append(uid)

        hybrid_ranked = []
        for uid in candidate_ids:
            vector_distance = vector_raw.get(uid)
            vector_score = _distance_to_similarity(vector_distance)
            bm25_score = bm25_scores.get(uid, 0.0)
            hybrid_score = self.vector_weight * vector_score + self.bm25_weight * bm25_score
            hybrid_ranked.append((uid, hybrid_score, vector_score, bm25_score, vector_distance))

        hybrid_ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        hybrid_ranked = hybrid_ranked[:n_results]

        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []
        hybrid_scores: List[float] = []
        vector_scores: List[float] = []
        bm25_out: List[float] = []
        vector_distances: List[Optional[float]] = []

        for uid, hybrid_score, vector_score, bm25_score, vector_distance in hybrid_ranked:
            entry = self.uid_to_entry.get(uid)
            ids.append(uid)
            metadatas.append(dict(vector_meta.get(uid) or (entry.metadata if entry else {})))
            documents.append(entry.document if entry else "")
            hybrid_scores.append(hybrid_score)
            vector_scores.append(vector_score)
            bm25_out.append(bm25_score)
            vector_distances.append(vector_distance)

        return HybridQueryResult(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            hybrid_scores=hybrid_scores,
            vector_scores=vector_scores,
            bm25_scores=bm25_out,
            vector_distances=vector_distances,
        )

    def _query_vector(
        self,
        *,
        query_embedding: Optional[List[float]],
        n_results: int,
        where_filter: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]]]:
        if self.collection is None or query_embedding is None:
            return {}, {}

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],
                where=where_filter,
            )
        except TypeError:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
        except Exception:
            return {}, {}

        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        vector_raw: Dict[str, Optional[float]] = {}
        vector_meta: Dict[str, Dict[str, Any]] = {}

        for index, uid in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] is not None else {}
            if not _metadata_matches(metadata, where_filter):
                continue

            distance = distances[index] if index < len(distances) else None
            vector_raw[str(uid)] = distance
            vector_meta[str(uid)] = dict(metadata)

        return vector_raw, vector_meta
