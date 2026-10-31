import unittest
from search import SearchIndex, evaluate, load_documents, ROOT, tokenize


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.index = SearchIndex(load_documents(ROOT / "data/corpus.json"))

    def test_arabic_diacritics_and_variants(self):
        self.assertEqual(tokenize("إِعَادَة أَلف آمال على"), tokenize("اعادة الف امال علي"))

    def test_exact_source_snippets(self):
        result = self.index.search("HTTP 429 exponential backoff")
        self.assertEqual(result["matches"][0]["source_id"], "api-rate-limit")
        documents = {doc["id"]: doc["text"] for doc in self.index.documents}
        for match in result["matches"]:
            self.assertIn(match["snippet"], documents[match["source_id"]])
            self.assertGreaterEqual(match["score"], result["threshold"])

    def test_unknown_empty_and_threshold(self):
        for query in ["", "123xyzunseen", "وصفة كنافة"]:
            self.assertEqual(self.index.search(query)["status"], "no_answer")
        self.assertEqual(self.index.search("password", threshold=1)["matches"], [])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            SearchIndex([])
        with self.assertRaises(ValueError):
            SearchIndex([{"id":"x","title":"a","text":"b"}]*2)
        for kwargs in [{"k":0}, {"threshold":float("nan")}, {"threshold":-1}]:
            with self.assertRaises(ValueError):
                self.index.search("password", **kwargs)

    def test_relevance_metrics_use_all_relevant_documents(self):
        index = SearchIndex([{"id":"a","title":"alpha","text":"topic"},
                             {"id":"b","title":"beta","text":"topic"}])
        report = evaluate(index, [{"id":"q","query":"topic", "relevant_ids":["a","b"]}], k=1)
        self.assertEqual(report["recall_at_k"], 0.5)
        self.assertEqual(report["mrr_at_k"], 1)

    def test_deterministic_tie_break_and_multilingual_query(self):
        self.assertEqual(self.index.search("تصدير CSV"), self.index.search("تصدير CSV"))
        self.assertEqual(self.index.search("تصدير CSV")["matches"][0]["source_id"], "export-csv")


if __name__ == "__main__":
    unittest.main()
