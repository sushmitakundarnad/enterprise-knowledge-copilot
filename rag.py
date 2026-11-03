
import hashlib

import chromadb
import fitz
import ollama

# Configuration
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="enterprise_documents",
    metadata={"hnsw:space": "cosine"}
)

def extract_pages(filename, file_bytes):
    """Extract text from a PDF or TXT file."""

    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            return [
                (number + 1, page.get_text())
                for number, page in enumerate(pdf)
            ]

    if filename.lower().endswith(".txt"):
        return [
            (1, file_bytes.decode("utf-8", errors="replace"))
        ]

    raise ValueError("Only PDF and TXT files are supported.")

def split_text(text, chunk_size=1000, overlap=150):
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

def embed_texts(texts):
    """Create embeddings with Ollama."""

    if not texts:
        return []

    result = ollama.embed(
        model=EMBED_MODEL,
        input=texts
    )

    return result["embeddings"]

def ingest_file(filename, file_bytes, session_id="default"):
    """Extract, embed, and index an uploaded document."""

    pages = extract_pages(filename, file_bytes)

    documents = []
    metadatas = []
    ids = []

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
                f"{session_id}:{filename}:"
                f"{page_number}:{chunk_number}:{chunk}"
            )

            ids.append(
                hashlib.sha256(
                    unique_text.encode("utf-8")
                ).hexdigest()
            )

    if not documents:
        raise ValueError(
            "No readable text found in this file. "
            "Scanned PDFs may require OCR."
        )

    # Generate embeddings before replacing an older indexed copy.
    embeddings = []

    try:
        for start in range(0, len(documents), 16):
            batch = documents[start:start + 16]
            embeddings.extend(embed_texts(batch))

    except Exception as error:
        raise RuntimeError(
            f"Embedding generation failed. Check that Ollama is "
            f"available and the '{EMBED_MODEL}' model is installed. "
            f"Details: {error}"
        ) from error

    if len(embeddings) != len(documents):
        raise RuntimeError("Embedding count does not match chunk count.")

    # Remove the previous version of this file in this chat.
    existing = collection.get(include=["metadatas"])
    old_ids = []

    for record_id, metadata in zip(
        existing.get("ids") or [],
        existing.get("metadatas") or []
    ):
        if (
            metadata
            and metadata.get("session_id") == session_id
            and metadata.get("source") == filename
        ):
            old_ids.append(record_id)

    if old_ids:
        collection.delete(ids=old_ids)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)

def answer_question(question, session_id="default"):
    """Retrieve relevant chunks and generate an answer."""

    try:
        question_embedding = embed_texts([question])[0]

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=4,
            where={"session_id": session_id},
            include=["documents", "metadatas", "distances"]
        )

    except Exception as error:
        raise RuntimeError(
            "Could not search indexed documents. "
            f"Details: {error}"
        ) from error

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    if not documents:
        return (
            "No documents have been indexed in this chat. "
            "Please upload a PDF or TXT file and click "
            "'Index Documents'.",
            []
        )

    passages = []

    for document, metadata in zip(documents, metadatas):
        passages.append({
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", 1)
        })

    context = "\n\n".join(
        f"Source: {item['source']}, Page: {item['page']}\n"
        f"{item['text']}"
        for item in passages
    )

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Enterprise Knowledge Copilot. "
                        "Answer using only the supplied document context. "
                        "Do not invent facts. If the answer is not present, "
                        "say that it was not found in the provided documents. "
                        "Use clear, simple English."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ],
            options={"temperature": 0.2}
        )

        answer = response["message"]["content"]

    except Exception as error:
        raise RuntimeError(
            f"Could not generate an answer. Check that Ollama is "
            f"available and '{LLM_MODEL}' is installed. Details: {error}"
        ) from error

    return answer, passages