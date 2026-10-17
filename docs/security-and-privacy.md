# Security and privacy

Manual excerpts, the user's question, and limited recent chat history are sent to OpenAI. During index construction, manual chunks are sent to the embeddings endpoint. The PDFs and FAISS indexes remain local.

The application does not log questions, answers, excerpts, API keys, or PDF content. Structured logs contain request ID, manual ID, stage timing, and token counts when returned. Keep `.env`, local PDFs, generated indexes, and detailed evaluation case reports out of source control; only the aggregate baseline summary is versioned.

PDF text is untrusted data. The generation prompt explicitly treats instructions inside excerpts as data. Citation IDs are checked against the retrieved context and selected manual before display. Only catalog paths within the repository root are accepted, and their hashes are validated.

The bundled manuals have no confirmed redistribution authorization. The Yamaha file includes an explicit copyright restriction. They are excluded from Git and Docker; users must obtain manuals legally and mount or copy them locally.

Managed environments that inspect TLS can set `MMA_CA_BUNDLE` to their trusted PEM bundle. The application keeps certificate verification enabled and does not provide an insecure bypass.
