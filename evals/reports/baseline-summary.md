# Initial live baseline

The first bounded live run evaluated all 48 human-checked cases with `text-embedding-3-small` and `gpt-4o-mini`. This is the pre-optimization baseline recorded on 2026-10-17.

The authorized index build produced 172 Yamaha chunks with fingerprint `b7ff828876973d8181a906d1` and 143 Voge chunks with fingerprint `cd54ea2c5205ff802301a918`; both use 1,536-dimensional vectors. The initial build did not retain embedding usage counters, so its exact API call and token totals are unavailable. Subsequent builds emit one structured event per embedding request and include provider token usage when returned.

| Measure | Result |
|---|---:|
| Retrieval cases with evidence labels | 41 |
| Recall@6 | 95.12% |
| MRR | 0.680 |
| Expected-status accuracy | 62.50% |
| Required-term accuracy | 64.58% |
| Citation-page accuracy among answered labeled cases | 79.17% |
| Pipeline errors | 13 |
| Cases with a critical failure | 20 |
| Input tokens reported | 67,237 |
| Output tokens reported | 1,623 |

Thirteen pipeline errors came from answers whose structured source IDs were valid but whose prose omitted the corresponding inline marker. Other visible failures included conditional-pressure ambiguity, Voge oil-capacity ambiguity, and follow-up retrieval without enough conversational context. These findings led to deterministic inline-marker completion, preflight clarification for known conditional questions, and contextual follow-up retrieval. Those changes have offline coverage but have not been measured in a second paid run.

The original live run was interrupted once and resumed from a case checkpoint. As a result, an exact end-to-end elapsed time for all 48 cases is unavailable. Per-case latency remains in the local detailed report. The finite sample and term-based checks do not establish universal semantic correctness.
