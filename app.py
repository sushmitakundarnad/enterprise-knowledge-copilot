
import uuid
from collections import defaultdict

import streamlit as st

from rag import (
    ingest_file,
    answer_question,
    preview_file_chunks
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Enterprise Knowledge Copilot",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_id = st.session_state.chat_id

# ==========================================
# MIDNIGHT BLUE UI
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #071426;
    color: #edf4ff;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1000px;
}

.brand {
    color: #edf4ff;
    font-size: 27px;
    font-weight: 800;
}

.hero {
    text-align: center;
    padding: 60px 10px 30px;
}

.hero h1 {
    font-size: 38px;
    color: #f4f7ff;
}

.hero p {
    color: #9aadc7;
    font-size: 16px;
}

.stButton button {
    background: #0d1d33;
    color: #ffffff;
    border: 1px solid #304867;
    border-radius: 12px;
    min-height: 45px;
}

.stButton button:hover {
    border-color: #7c9cff;
    color: #ffffff;
}

[data-testid="stChatMessage"] {
    background: #0d1d33;
    border: 1px solid #233b59;
    border-radius: 15px;
}

[data-testid="stFileUploader"] {
    background: #0d1d33;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #7187a6;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER AND NEW CHAT
# ==========================================

left, right = st.columns([5, 1])

with left:
    st.markdown(
        '<div class="brand">📚 Knowledge Copilot</div>',
        unsafe_allow_html=True
    )
    st.caption("Your documents. Your answers. Your knowledge.")

with right:
    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.chat_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

st.divider()

chat_id = st.session_state.chat_id

# ==========================================
# UPLOAD AND INDEX DOCUMENTS
# ==========================================

with st.expander("📂 Upload Documents", expanded=False):

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{chat_id}"
    )

    # Show the estimated chunk count inside the uploader.
    if uploaded_files:
        st.markdown("### Selected documents")

        for file in uploaded_files:
            try:
                chunk_count = preview_file_chunks(
                    file.name,
                    file.getvalue()
                )

                st.caption(
                    f"📄 {file.name} — approximately "
                    f"{chunk_count} text chunks"
                )

            except Exception as error:
                st.warning(
                    f"Could not preview {file.name}: {error}"
                )

    if st.button(
        "Index Documents",
        key=f"index_{chat_id}",
        use_container_width=True
    ):

        if not uploaded_files:
            st.warning("Please upload at least one PDF or TXT file.")

        else:
            successful = 0
            total_files = len(uploaded_files)
            progress = st.progress(0)

            for index, file in enumerate(uploaded_files):
                st.write(f"Processing: **{file.name}**")

                try:
                    chunks_created = ingest_file(
                        filename=file.name,
                        file_bytes=file.getvalue(),
                        session_id=chat_id
                    )

                    st.success(
                        f"✅ {file.name} indexed successfully. "
                        f"Chunks created: {chunks_created}"
                    )

                    successful += 1

                except Exception as error:
                    st.error(
                        f"❌ Could not index {file.name}: {error}"
                    )

                progress.progress((index + 1) / total_files)

            if successful:
                st.session_state.messages = []

                st.success(
                    f"Indexing complete: {successful} of "
                    f"{total_files} file(s) indexed."
                )

# ==========================================
# WELCOME SCREEN
# ==========================================

if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <h1>What would you like to know?</h1>
        <p>
        Upload your documents and ask questions.
        Your AI assistant will find answers from your files.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Upload your documents, click Index Documents, "
        "and type a question below to get started."
    )

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

def display_sources(sources):
    grouped = defaultdict(set)

    for source in sources:
        grouped[source["source"]].add(source["page"])

    if grouped:
        with st.expander("📚 View Sources"):
            for filename, pages in sorted(grouped.items()):
                st.markdown(f"📄 **{filename}**")

                page_numbers = ", ".join(
                    str(page) for page in sorted(pages)
                )

                st.caption(f"Pages: {page_numbers}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))

# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input("Ask about your documents...")

# ==========================================
# GENERATE ANSWER
# ==========================================

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                answer, sources = answer_question(
                    question=question,
                    session_id=chat_id
                )

            st.markdown(answer)
            display_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:
            st.error(f"Answer generation failed: {error}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Sorry, I couldn't generate an answer. "
                    "Please check the error displayed above."
                ),
                "sources": []
            })

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    Powered by Python · Streamlit · Ollama · ChromaDB · RAG
</div>
""", unsafe_allow_html=True)