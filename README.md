# Support Search — bilingual evidence retrieval

An independent NLP engineering demo: a small, auditable search engine
over English/Arabic support documentation. It returns ranked source IDs and exact
source snippets, or an explicit `no_answer`. There is no LLM, answer generation,
network access, pretrained model download or external dependency.

## Run

Requires Python 3.10+. Run from this directory:

```bash
python search.py index
python search.py search "HTTP 429 exponential backoff"
python search.py search "إعادة كلمة المرور"
python search.py evaluate --output examples/evaluation.json
python -m unittest -v
```

`--corpus`, `--queries`, `--k` and `--threshold` expose input and ranking choices.
For example, `python search.py search "unrelated astronomy" --threshold 0.3`.
UTF-8 files preserve Arabic independently of terminal encoding.

## Design and engineering decisions

1. Unicode case folding, Arabic diacritic removal and common letter normalization
   support mixed-language text. This is not stemming, translation or semantic
   cross-language matching: English queries require English vocabulary in sources.
2. Sublinear term frequency and smoothed inverse-document frequency reduce repeated
   token dominance. L2-normalized vectors make dot products cosine similarities.
3. A fixed 0.18 score threshold rejects weak or absent matches. This score is **not
   a confidence probability**. Common-term false positives remain possible.
4. Stable ID tie-breaking provides reproducible rankings. Exact sentence snippets
   allow clients to inspect evidence and avoid fabricated generated answers.
5. The fixture evaluator reports mean Recall@k over answerable queries, MRR@k,
   and rejection accuracy on unanswerable queries separately. Unknown queries
   cannot inflate answerable recall.

## Data and results

`data/corpus.json` contains ten original fictional support documents. No customer
records, private information, scraped text or borrowed policies are used.
`data/queries.json` contains 18 separate authored query/relevance records, including
four explicit out-of-domain negatives. The IDF is computed from documents only;
queries do not enter index construction. The threshold is a fixed engineering
default, not optimized on this evaluation file.

The actual run is checked in as `examples/evaluation.json`; `examples/search.json`
shows a bilingual query result. This deliberately easy, tiny authored fixture is
a reproducibility demonstration, **not a production accuracy benchmark**. See
MODEL_CARD.md before using these results in a proposal.

## Extending responsibly

Use a separate representative validation set to choose thresholds, then freeze
them before measuring an independent test set. Add paraphrases, ambiguous and
near-domain negative queries, typo/noise tests and language-specific slices.
Evaluate an embedding retriever against this lexical baseline before accepting
its extra cost. Production use also needs document access controls, PII handling,
update/version policy, latency measurements and monitoring.
