import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_document_ids" not in st.session_state:
    st.session_state.selected_document_ids = set()


def upload_document(file) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/documents",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
    )
    response.raise_for_status()
    return response.json()


def fetch_documents() -> list[dict]:
    response = requests.get(f"{API_BASE_URL}/documents")
    response.raise_for_status()
    return response.json()


def ask_question(question: str, document_ids: list[str] | None = None) -> dict:
    params = {"question": question, "session_id": st.session_state.session_id}
    if document_ids:
        params["document_ids"] = document_ids
    response = requests.post(
        f"{API_BASE_URL}/chat",
        params=params,
    )
    response.raise_for_status()
    return response.json()


STATUS_ICON = {"processing": "⏳", "ready": "✅", "failed": "❌"}


def format_source(source: dict) -> str:
    page = source.get("page_number")
    page_part = f", page {page}" if page is not None else ""
    snippet = (source.get("text") or "").strip().replace("\n", " ")
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    return f"**Document {source.get('document_id', '?')}{page_part}**\n\n{snippet}"

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if st.button("Upload", type="primary", disabled=uploaded_file is None):
        try:
            with st.spinner("Uploading..."):
                result = upload_document(uploaded_file)
            if result.get("duplicate"):
                st.info(f"'{result['filename']}' already exists (status: {result['status']}).")
            else:
                st.success(f"Uploaded '{result['filename']}' — processing started.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    st.divider()

    header_col, refresh_col = st.columns([3, 1])
    header_col.header("Documents")
    if refresh_col.button("↻", help="Refresh document list"):
        st.rerun()

    try:
        documents = fetch_documents()
    except requests.RequestException as exc:
        documents = []
        st.error(f"Could not load documents: {exc}")

    if not documents:
        st.caption("No documents uploaded yet.")
    else:
        ready_ids = {doc["id"] for doc in documents if doc["status"] == "ready"}
        st.session_state.selected_document_ids &= ready_ids
        st.caption("Select documents to search. None selected = search all.")
        for doc in documents:
            icon = STATUS_ICON.get(doc["status"], "•")
            with st.container(border=True):
                caption = f"Status: {doc['status']}"
                if doc.get("num_chunks"):
                    caption += f" · {doc['num_chunks']} chunks"
                if doc["status"] == "ready":
                    checked = st.checkbox(
                        f"{icon} {doc['filename']}",
                        value=doc["id"] in st.session_state.selected_document_ids,
                        key=f"doc_select_{doc['id']}",
                    )
                    if checked:
                        st.session_state.selected_document_ids.add(doc["id"])
                    else:
                        st.session_state.selected_document_ids.discard(doc["id"])
                else:
                    st.markdown(f"**{icon} {doc['filename']}**")
                st.caption(caption)

st.title("📄 PDF Chatbot")
st.caption("Ask questions about your uploaded documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(format_source(source))
                    st.divider()

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = ask_question(
                    question, document_ids=list(st.session_state.selected_document_ids)
                )
            st.session_state.session_id = result["session_id"]
            answer = result["answer"]
            sources = result.get("sources") or []
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(format_source(source))
                        st.divider()
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except requests.RequestException as exc:
            error_message = f"Error: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
