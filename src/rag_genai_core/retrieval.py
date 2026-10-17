from __future__ import annotations

from typing import List, Sequence

import faiss
import numpy as np

from .schemas import Chunk


class ManualRetriever:
    def __init__(self, index, chunks: Sequence[Chunk], embed_query, k: int = 6):
        self.index = index
        self.chunks = list(chunks)
        self.embed_query = embed_query
        self.k = k

    def retrieve(self, question: str) -> List[Chunk]:
        vector = np.asarray([self.embed_query(question)], dtype="float32")
        faiss.normalize_L2(vector)
        _, positions = self.index.search(vector, min(self.k, len(self.chunks)))
        return [self.chunks[int(position)] for position in positions[0] if position >= 0]
