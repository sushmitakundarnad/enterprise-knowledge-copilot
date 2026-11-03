
import hashlib
import re

import chromadb
import fitz
import ollama

# ==========================================
# CONFIGURATION
# ==========================================

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="enterprise_documents",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# EXTRACT TEXT FROM PDF OR TXT
# ==========================================

def extract_pages(filename, file_bytes):
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            return [
                (page_number + 1, page.get_text())
                for page_number, page in enumerate(pdf)
            ]

    if filename.lower().endswith(".txt"):
        return [
            (1, file_bytes.decode("utf-8", errors="replace"))
        ]

    raise ValueError("Only PDF and TXT files are supported.")

# ==========================================
# SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text(text, chunk_size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    step = chunk_size - overlap

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
        if text[start:start + chunk_size].strip()
    ]

# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

def embed_texts(texts):
    if not texts:
        return []

    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts
    )

    return response["embeddings"]

# ==========================================
# PREVIEW CHUNK COUNT BEFORE INDEXING
# ==========================================

def preview_file_chunks(filename, file_bytes):
    pages = extract_pages(filename, file_bytes)

    return sum(
        len(split_text(page_text))
        for _, page_text in pages
    )

# ==========================================
# INDEX A PDF OR TXT FILE
# ==========================================

def ingest_file(filename, file_bytes, session_id="default"):
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

            chunk_id = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

    if not documents:
        raise ValueError(
            "No readable text was found in this file. "
            "Scanned PDFs may require OCR."
        )

    # Generate embeddings before changing the existing index.
    embeddings = []

    try:
        for start in range(0, len(documents), 16):
            batch = documents[start:start + 16]
            embeddings.extend(embed_texts(batch))

    except Exception as error:
        raise RuntimeError(
            f"Could not generate embeddings. Make sure Ollama "
            f"is running and '{EMBED_MODEL}' is installed. "
            f"Details: {error}"
        ) from error

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "The number of embeddings does not match the chunks."
        )

    # Find an older version of this file in the same chat.
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
            and metadata.get("session_id") == session_id
            and metadata.get("source") == filename
        ):
            old_ids.append(record_id)

    # Replace the old version of this file.
    if old_ids:
        collection.delete(ids=old_ids)

    # Store the new chunks and embeddings.
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)

# ==========================================
# RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve(question, session_id="default", top_k=4):
    # Check whether this chat has indexed text.
    existing = collection.get(
        include=["metadatas"]
    )

    session_ids = [
        record_id
        for record_id, metadata in zip(
            existing.get("ids") or [],
            existing.get("metadatas") or []
        )
        if metadata and metadata.get("session_id") == session_id
    ]

    if not session_ids:
        return []

    # Embed the question.
    question_embedding = embed_texts([question])[0]

    # Search for relevant chunks from this chat only.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, len(session_ids)),
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    passages = []

    for document, metadata, distance in zip(
        documents[0], metadatas[0], distances[0]
    ):
        passages.append({
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", 1),
            "similarity": 1 - distance
        })

    return passages

# ==========================================
# ANSWER QUESTIONS USING RAG
# ==========================================

def answer_question(question, session_id="default"):
    passages = retrieve(
        question=question,
        session_id=session_id,
        top_k=4
    )

    if not passages:
        return (
            "No documents have been indexed in this chat. "
            "Please upload a PDF or TXT file and click "
            "'Index Documents'.",
            []
        )

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
                        "Do not invent facts. If the answer is not in "
                        "the context, clearly say that the information "
                        "was not found in the provided documents. "
                        "Use simple, clear English."
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
            f"Could not generate an answer. Make sure Ollama "
            f"is running and '{LLM_MODEL}' is installed. "
            f"Details: {error}"
        ) from error

    return answer, passages