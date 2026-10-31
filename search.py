"""Dependency-free, source-grounded bilingual support retrieval.

This module ranks authored documents; it never generates an answer that is absent
from them. It is deliberately small enough to audit and reproduce on a CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLD = 0.18
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
ARABIC_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ـ": ""})


def tokenize(text: str) -> list[str]:
    """Normalize case/Arabic variants without claiming stemming or translation."""
    # Diacritics should not prevent an Arabic query matching unvocalized text.
    text = "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))
    return TOKEN_RE.findall(text.translate(ARABIC_MAP).casefold())


def normalize(vector: dict[str, float]) -> dict[str, float]:
    length = math.sqrt(sum(value * value for value in vector.values()))
    return {key: value / length for key, value in vector.items()} if length else {}


class SearchIndex:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError("At least one document is required")
        ids = [doc.get("id") for doc in documents]
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("Document IDs must be nonempty and unique")
        for doc in documents:
            if not isinstance(doc.get("title"), str) or not isinstance(doc.get("text"), str):
                raise ValueError("Each document needs title and text strings")
        self.documents = documents
        counts = [Counter(tokenize(doc["title"] + " " + doc["text"])) for doc in documents]
        frequency = Counter(token for count in counts for token in count)
        # Smoothed IDF avoids division by zero and supports tiny corpora.
        self.idf = {word: math.log((1 + len(documents)) / (1 + df)) + 1
                    for word, df in frequency.items()}
        self.vectors = [self._vector(count) for count in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        return normalize({word: (1 + math.log(count)) * self.idf[word]
                          for word, count in counts.items() if word in self.idf})

    def search(self, query: str, k: int = 3, threshold: float = DEFAULT_THRESHOLD) -> dict:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        vector = self._vector(Counter(tokenize(query)))
        ranked = []
        for document, indexed in zip(self.documents, self.vectors):
            score = sum(weight * indexed.get(word, 0) for word, weight in vector.items())
            if score > 0 and score >= threshold:
                # Snippets are exact document substrings, not paraphrased answers.
                sentences = re.split(r"(?<=[.!؟])\s+", document["text"])
                snippet = max(sentences, key=lambda s: sum(t in vector for t in tokenize(s)))
                ranked.append({"source_id": document["id"], "title": document["title"],
                               "score": round(score, 6), "snippet": snippet})
        ranked.sort(key=lambda item: (-item["score"], item["source_id"]))
        return {"query": query, "status": "matched" if ranked else "no_answer",
                "threshold": threshold, "matches": ranked[:k]}


def load_documents(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["documents"]


def evaluate(index: SearchIndex, queries: list[dict], k: int = 3,
             threshold: float = DEFAULT_THRESHOLD) -> dict:
    if not queries:
        raise ValueError("Evaluation queries are required")
    answerable, unanswerable, recalls, reciprocals, correct_rejections, rows = 0, 0, [], [], 0, []
    known_ids = {doc["id"] for doc in index.documents}
    for item in queries:
        relevant = set(item["relevant_ids"])
        if not relevant <= known_ids:
            raise ValueError("Relevance labels reference unknown document IDs")
        result = index.search(item["query"], k, threshold)
        found = [match["source_id"] for match in result["matches"]]
        if relevant:
            answerable += 1
            recall = len(relevant.intersection(found)) / len(relevant)
            rr = next((1 / rank for rank, source in enumerate(found, 1) if source in relevant), 0)
            recalls.append(recall)
            reciprocals.append(rr)
        else:
            unanswerable += 1
            correct_rejections += result["status"] == "no_answer"
        rows.append({"id": item["id"], "query": item["query"],
                     "relevant_ids": sorted(relevant), "retrieved_ids": found,
                     "status": result["status"]})
    return {"dataset": "Author-written bilingual support fixture v1",
            "evaluation_type": "small curated regression fixture; not an independent benchmark",
            "k": k, "threshold": threshold, "answerable_queries": answerable,
            "unanswerable_queries": unanswerable,
            "recall_at_k": sum(recalls) / answerable if answerable else None,
            "mrr_at_k": sum(reciprocals) / answerable if answerable else None,
            "no_answer_accuracy": correct_rejections / unanswerable if unanswerable else None,
            "queries": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "evaluate", "index"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.json")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    index = SearchIndex(load_documents(args.corpus))
    if args.command == "evaluate":
        result = evaluate(index, json.loads(args.queries.read_text(encoding="utf-8"))["queries"], args.k, args.threshold)
    elif args.command == "index":
        result = {"documents": len(index.documents), "vocabulary": len(index.idf),
                  "source_ids": [doc["id"] for doc in index.documents]}
    else:
        result = index.search(args.query, args.k, args.threshold)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
