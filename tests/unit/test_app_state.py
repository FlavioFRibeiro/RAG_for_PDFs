from apps.streamlit_app import clear_manual_context


def test_changing_manual_clears_history_sources_and_active_context() -> None:
    state = {
        "messages": [
            {
                "role": "assistant",
                "content": "answer",
                "sources": [{"chunk_id": "old-manual-chunk"}],
            }
        ],
        "active_manual_id": "old-manual",
    }
    clear_manual_context(state)
    assert state == {"messages": [], "active_manual_id": None}
