# Evaluation method

The evaluation set contains 48 human-checked cases, split into 32 development and 16 reserved cases with equal representation for each motorcycle. Evidence labels use physical PDF pages. It covers conditional specifications, procedures, follow-ups, absence, ambiguity, wrong-motorcycle questions, isolation, and adversarial instructions.

Retrieval reports Recall@k and reciprocal rank only for cases with evidence-page labels. Generation reports status correctness, required numeric/unit terms, citation pages, latency, and token counts. Critical case failures remain visible in the report instead of being hidden by averages. Semantic support and nuanced procedure quality still require human review.

Offline tests use fake embeddings and generation and require no API key. Live retrieval and generation evaluations require explicit execution because they incur OpenAI usage. Results depend on model versions and the local corpus; they are not a guarantee of accuracy.

## Initial measured baseline

The authorized 48-case live run used `text-embedding-3-small`, `gpt-4o-mini`, retrieval `k=6`, and the initial implementation. Recall@6 was 95.12% and MRR was 0.680 over 41 cases with evidence labels. Expected-status accuracy was 62.50%, required-term accuracy was 64.58%, and citation-page accuracy was 79.17% among answered cases with evidence labels. The run reported 67,237 input tokens and 1,623 output tokens.

There were 13 pipeline errors and 20 cases with at least one critical failure. The pipeline errors primarily exposed valid structured source IDs that were omitted from the answer prose. The baseline also exposed conditional-pressure ambiguity, the two Voge oil quantities, and weak follow-up retrieval. The implementation now covers those failure modes deterministically, but no second paid evaluation has been run. See the tracked [baseline summary](../evals/reports/baseline-summary.md); the detailed per-case report remains a local generated artifact.
