import hashlib
import os

import chromadb
import fitz
import streamlit as st

from google import genai
from google.genai import types


# ==========================================
# GEMINI CONFIGURATION
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Add it to Streamlit Secrets or your local environment."
    )


gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# Text generation model
LLM_MODEL = "gemini-3.5-flash-lite"

# Text embedding model
EMBED_MODEL = "gemini-embedding-001"


# ==========================================
# CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# New collection name because the old collection
# contains Ollama/Nomic embeddings with a different
# embedding space.
collection = client.get_or_create_collection(
    name="enterprise_documents_gemini",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ==========================================
# EXTRACT TEXT
# ==========================================

def extract_pages(filename, file_bytes):
    """Extract text from PDF or TXT."""

    if filename.lower().endswith(".pdf"):

        with fitz.open(
            stream=file_bytes,
            filetype="pdf"
        ) as pdf:

            return [
                (
                    number + 1,
                    page.get_text()
                )
                for number, page in enumerate(pdf)
            ]

    if filename.lower().endswith(".txt"):

        return [
            (
                1,
                file_bytes.decode(
                    "utf-8",
                    errors="replace"
                )
            )
        ]

    raise ValueError(
        "Only PDF and TXT files are supported."
    )


# ==========================================
# TEXT CHUNKING
# ==========================================

def split_text(
    text,
    chunk_size=1000,
    overlap=150
):
    """Split text into overlapping chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    step = chunk_size - overlap

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
        if text[start:start + chunk_size].strip()
    ]


# ==========================================
# GEMINI EMBEDDINGS
# ==========================================

def embed_documents(texts):
    """
    Generate embeddings for document chunks
    using Gemini Embedding.
    """

    if not texts:
        return []

    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


def embed_query(question):
    """
    Generate an embedding for the user's question.
    """

    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return result.embeddings[0].values


# ==========================================
# INGEST DOCUMENT
# ==========================================

def ingest_file(
    filename,
    file_bytes,
    session_id="default"
):
    """Extract, chunk, embed, and index a document."""

    pages = extract_pages(
        filename,
        file_bytes
    )

    documents = []
    metadatas = []
    ids = []

    # --------------------------------------
    # Create chunks
    # --------------------------------------

    for page_number, page_text in pages:

        chunks = split_text(page_text)

        for chunk_number, chunk in enumerate(chunks):

            documents.append(chunk)

            metadatas.append({
                "source": filename,
                "page": page_number,
                "session_id": session_id
            })

            unique_text = (
                f"{session_id}:"
                f"{filename}:"
                f"{page_number}:"
                f"{chunk_number}:"
                f"{chunk}"
            )

            ids.append(
                hashlib.sha256(
                    unique_text.encode("utf-8")
                ).hexdigest()
            )

    # --------------------------------------
    # Check document
    # --------------------------------------

    if not documents:

        raise ValueError(
            "No readable text found in this file. "
            "Scanned PDFs may require OCR."
        )

    # --------------------------------------
    # Generate Gemini embeddings
    # --------------------------------------

    embeddings = []

    try:

        batch_size = 16

        for start in range(
            0,
            len(documents),
            batch_size
        ):

            batch = documents[
                start:start + batch_size
            ]

            batch_embeddings = embed_documents(
                batch
            )

            embeddings.extend(
                batch_embeddings
            )

    except Exception as error:

        raise RuntimeError(
            "Gemini embedding generation failed. "
            "Please check your GEMINI_API_KEY "
            "and Gemini API access. "
            f"Details: {error}"
        ) from error

    # --------------------------------------
    # Verify embedding count
    # --------------------------------------

    if len(embeddings) != len(documents):

        raise RuntimeError(
            "Embedding count does not match "
            "document chunk count."
        )

    # --------------------------------------
    # Remove previous copy of same file
    # --------------------------------------

    existing = collection.get(
        include=["metadatas"]
    )

    old_ids = []

    for record_id, metadata in zip(
        existing.get("ids") or [],
        existing.get("metadatas") or []
    ):

        if (
            metadata
            and metadata.get("session_id")
            == session_id
            and metadata.get("source")
            == filename
        ):

            old_ids.append(record_id)

    if old_ids:

        collection.delete(
            ids=old_ids
        )

    # --------------------------------------
    # Store in ChromaDB
    # --------------------------------------

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


# ==========================================
# ANSWER QUESTION
# ==========================================

def answer_question(
    question,
    session_id="default"
):
    """Retrieve relevant chunks and generate an answer."""

    # --------------------------------------
    # Embed user question
    # --------------------------------------

    try:

        question_embedding = embed_query(
            question
        )

        results = collection.query(
            query_embeddings=[
                question_embedding
            ],
            n_results=4,
            where={
                "session_id": session_id
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    except Exception as error:

        raise RuntimeError(
            "Could not search indexed documents "
            "using Gemini embeddings. "
            f"Details: {error}"
        ) from error

    # --------------------------------------
    # Retrieve documents
    # --------------------------------------

    documents = (
        results.get("documents") or [[]]
    )[0]

    metadatas = (
        results.get("metadatas") or [[]]
    )[0]

    if not documents:

        return (
            "No documents have been indexed "
            "in this chat. Please upload a PDF "
            "or TXT file and click "
            "'Index Documents'.",
            []
        )

    # --------------------------------------
    # Prepare retrieved passages
    # --------------------------------------

    passages = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        passages.append({
            "text": document,
            "source": metadata.get(
                "source",
                "Unknown"
            ),
            "page": metadata.get(
                "page",
                1
            )
        })

    # --------------------------------------
    # Build RAG context
    # --------------------------------------

    context = "\n\n".join(
        f"Source: {item['source']}, "
        f"Page: {item['page']}\n"
        f"{item['text']}"
        for item in passages
    )

    # --------------------------------------
    # Gemini answer generation
    # --------------------------------------

    try:

        prompt = f"""
You are an Enterprise Knowledge Copilot.

Answer the user's question using ONLY the
provided document context.

Do not invent facts.

If the answer is not present in the
provided context, clearly say that the
answer was not found in the provided
documents.

Use clear and simple English.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

        response = gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )

        answer = response.text

    except Exception as error:

        raise RuntimeError(
            "Could not generate an answer using "
            "Gemini. Please check your Gemini API "
            f"configuration. Details: {error}"
        ) from error

    return answer, passages