# ADR 0001: Separate FAISS index per manual

Status: accepted

Use a distinct FAISS index and JSON chunk store for every catalog manual. This makes motorcycle isolation structural, keeps loading simple, and avoids relying on a runtime metadata filter. The small controlled corpus makes the extra files negligible.
