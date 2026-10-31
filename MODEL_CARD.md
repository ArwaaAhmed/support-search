# Retrieval model card

- **System:** TF-IDF lexical retrieval with cosine ranking and abstention.
- **Intended use:** inspectable support-document search demo and baseline.
- **Excluded use:** autonomous financial/medical/legal advice, production helpdesk
  claims, access-controlled document search without authorization controls.
- **Languages:** Arabic and English surface forms present in the corpus; mixed
  queries supported. No cross-language understanding is claimed.
- **Fit data:** ten authored fictional documents; no supervised training.
- **Evaluation:** separate hand-authored relevance file, 14 answerable and four
  unanswerable queries. This is an internal regression fixture and too small for
  external generalization claims.
- **Metrics:** Recall@3 is the fraction of relevant sources in top three, averaged
  over answerable queries. MRR@3 uses reciprocal first relevant rank (zero if
  missed). No-answer accuracy is correct rejections divided by negative queries.
- **Threshold:** fixed cosine 0.18, not calibrated as a probability or fitted to
  this fixture. In-domain unanswerable queries can share words and fool it.
- **Known limitations:** lexical paraphrase failure, tokenization simplicity,
  Arabic dialect/morphology gaps, no typo correction, document-length effects,
  exact snippets without entailment verification, no ACL or persistent index.
- **Compute:** CPU only; Python standard library. No GPU or API cost.
- **Provenance:** authored demo data and generated run reports; no real client,
  deployment or revenue claim.
