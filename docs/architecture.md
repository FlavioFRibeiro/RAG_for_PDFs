# Architecture

```mermaid
flowchart LR
    A[Maintainer catalog] --> B[Catalog validation]
    P[Local PDF manuals] --> B
    B --> C[Page text and table extraction]
    C --> D[Stable page chunks]
    D --> E[OpenAI embeddings]
    E --> F[One FAISS index per manual]
    U[Streamlit user] --> S[Motorcycle selection]
    S --> R[Selected manual retriever]
    F --> R
    R --> G[Grounded structured generation]
    G --> V[Citation validation]
    V --> U
```

The catalog is the source of truth. Indexes are separated by `manual_id`, so retrieval cannot cross motorcycle boundaries. Each index directory is addressed by a fingerprint of the PDF hash, parser and table rules, chunk settings, and embedding model. The application loads only a complete, compatible directory.

PyMuPDF supplies ordered text, table cells, page rendering, and one-page PDF exports. Tables are appended to page text in a row-preserving representation before page-local chunking. FAISS stores normalized vectors; chunk metadata is JSON. No pickle is loaded.

The core package has no Streamlit dependency. UI session state owns the limited conversation history and discards it whenever the motorcycle changes.

`pdf_preview.py` binds each source preview to the answer text citing that source and matches complete quantities against its retrieved evidence and the PDF text. It uses directional text quads for in-memory highlight annotations, including rotated pages. The full original page is always rendered without cropping, table detection, or saving PDF changes. Missing matches leave the page unmarked; short verbatim phrase matching supports non-numeric procedures. Streamlit caches the rendered image and highlight count together, including the catalog hash in its cache key.
