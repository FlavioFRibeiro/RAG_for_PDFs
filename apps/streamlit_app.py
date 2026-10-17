from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, MutableMapping

import fitz
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_genai_core.catalog import load_catalog, validate_catalog
from rag_genai_core.config import Settings
from rag_genai_core.exceptions import ManualAssistantError
from rag_genai_core.observability import configure_logging
from rag_genai_core.runtime import build_runtime
from rag_genai_core.schemas import ChatTurn

load_dotenv(ROOT / ".env")
configure_logging()
SETTINGS = Settings.from_env()


@st.cache_resource(show_spinner=False)
def cached_runtime(manual_id: str):
    return build_runtime(manual_id, SETTINGS)


@st.cache_data(show_spinner=False)
def render_page(pdf_path: str, physical_page: int, dpi: int) -> bytes:
    with fitz.open(pdf_path) as document:
        page = document.load_page(physical_page - 1)
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")


@st.cache_data(show_spinner=False)
def single_page_pdf(pdf_path: str, physical_page: int) -> bytes:
    with fitz.open(pdf_path) as source:
        output = fitz.open()
        output.insert_pdf(source, from_page=physical_page - 1, to_page=physical_page - 1)
        data = output.tobytes()
        output.close()
        return data


def clear_manual_context(state: MutableMapping[str, object]) -> None:
    state["messages"] = []
    state["active_manual_id"] = None


def reset_conversation() -> None:
    clear_manual_context(st.session_state)


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_manual_id", "")
    st.session_state.setdefault("active_manual_id", None)


def render_sources(sources: List[Dict[str, object]], record, key_prefix: str) -> None:
    for source in sources:
        printed = f"; printed page {source['printed_page']}" if source.get("printed_page") else ""
        label = f"[{source['source_id']}] PDF page {source['physical_page']}{printed}"
        with st.expander(label):
            st.code(source["evidence"], language=None)
            page_number = int(source["physical_page"])
            st.image(render_page(str(record.pdf_path), page_number, SETTINGS.page_dpi), use_column_width=True)
            st.download_button(
                "Open original page",
                data=single_page_pdf(str(record.pdf_path), page_number),
                file_name=f"{record.manual_id}-page-{page_number}.pdf",
                mime="application/pdf",
                key=f"page-{key_prefix}-{source['chunk_id']}",
            )


def main() -> None:
    st.set_page_config(page_title="Motorcycle Manual Assistant", page_icon="🏍️", layout="wide")
    initialize_state()
    st.title(" 🏍️ Motorcycle Manual Assistant 🔧")
    st.caption(
        "Answers use the available manual edition. Verify critical specifications in the cited source."
    )

    try:
        catalog = load_catalog(SETTINGS.catalog_path, SETTINGS.root_dir)
        validate_catalog(catalog.values())
    except ManualAssistantError as exc:
        st.error(str(exc))
        st.stop()

    choices = [""] + list(catalog)
    labels = {"": "Select your motorcycle"}
    labels.update({manual_id: record.display_name for manual_id, record in catalog.items()})
    selected = st.sidebar.selectbox(
        "Motorcycle",
        choices,
        format_func=lambda value: labels[value],
        key="selected_manual_id",
        on_change=reset_conversation,
    )
    if not selected:
        st.info("Please select a motorcycle model to begin.")
        st.stop()

    record = catalog[selected]
    st.sidebar.subheader(record.display_name)
    st.sidebar.write(record.title)
    st.sidebar.caption(
        f"Edition: {record.edition} · Model year: {record.model_year} · Market: {record.market}"
    )
    st.info(f"Current manual: **{record.title}**")

    examples = {
        "yamaha-tenere-700-xtz690d-2022-04-en": [
            "What is the drive chain slack specification?",
            "What are the cold tire pressures for off-road riding?",
            "What is the fuel tank capacity?",
        ],
        "voge-500ds-en-unknown-edition": [
            "What are the tire pressures for a rider and passenger?",
            "What is the fuel tank capacity?",
            "What is the engine oil capacity?",
        ],
    }
    with st.sidebar.expander("Example questions"):
        for example in examples[selected]:
            st.write(f"• {example}")

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("conditions"):
                st.caption("Conditions: " + "; ".join(message["conditions"]))
            if message.get("sources"):
                render_sources(message["sources"], record, f"history-{message_index}")

    question = st.chat_input("Ask a question about the selected motorcycle manual")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not configured. Add it to .env before asking questions.")
            st.session_state.messages.pop()
            return
        try:
            with st.spinner("Checking the manual..."):
                pipeline = cached_runtime(selected)
                history = [ChatTurn(item["role"], item["content"]) for item in st.session_state.messages[:-1]]
                response = pipeline.ask(question, history)
            st.markdown(response.answer)
            if response.conditions:
                st.caption("Conditions: " + "; ".join(response.conditions))
            serialized_sources = [asdict(source) for source in response.sources]
            if serialized_sources:
                render_sources(serialized_sources, record, response.request_id)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "status": response.status.value,
                    "conditions": response.conditions,
                    "sources": serialized_sources,
                }
            )
            st.session_state.active_manual_id = selected
        except ManualAssistantError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()


if __name__ == "__main__":
    main()
