
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
# GET INDEXED RECORDS FOR ONE CHAT
# ==========================================

def get_chat_records(session_id):
    records = collection.get(
        include=["documents", "metadatas"]
    )

    matching_ids = []
    matching_documents = []
    matching_metadatas = []

    for record_id, document, metadata in zip(
        records.get("ids") or [],
        records.get("documents") or [],
        records.get("metadatas") or []
    ):
        if metadata and metadata.get("session_id") == session_id:
            matching_ids.append(record_id)
            matching_documents.append(document)
            matching_metadatas.append(metadata)

    return {
        "ids": matching_ids,
        "documents": matching_documents,
        "metadatas": matching_metadatas
    }

# ==========================================
# COUNT DOCUMENTS AND CHUNKS
# ==========================================

def get_chat_index_summary(session_id):
    records = get_chat_records(session_id)

    filenames = {
        metadata.get("source")
        for metadata in records["metadatas"]
        if metadata.get("source")
    }

    return {
        "files": len(filenames),
        "chunks": len(records["ids"])
    }

def get_chat_document_count(session_id):
    return get_chat_index_summary(session_id)["chunks"]

# ==========================================
# PREVIEW CHUNK COUNT
# ==========================================

def preview_file_chunks(filename, file_bytes):
    pages = extract_pages(filename, file_bytes)

    return sum(
        len(split_text(page_text))
        for _, page_text in pages
    )

# ==========================================
# INDEX AN UPLOADED FILE
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

            ids.append(
                hashlib.sha256(
                    unique_text.encode("utf-8")
                ).hexdigest()
            )

    if not documents:
        raise ValueError(
            "No readable text was found in this file. "
            "Scanned PDFs may require OCR."
        )

    # Generate embeddings in batches.
    embeddings = []

    try:
        for start in range(0, len(documents), 16):
            batch = documents[start:start + 16]
            embeddings.extend(embed_texts(batch))

    except Exception as error:
        raise RuntimeError(
            f"Could not generate embeddings. Check that Ollama "
            f"is running and '{EMBED_MODEL}' is installed. "
            f"Details: {error}"
        ) from error

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "The number of embeddings does not match the chunks."
        )

    # Remove an older version of the same file in this chat.
    existing = get_chat_records(session_id)

    old_ids = [
        record_id
        for record_id, metadata in zip(
            existing["ids"], existing["metadatas"]
        )
        if metadata.get("source") == filename
    ]

    if old_ids:
        collection.delete(ids=old_ids)

    # Save the new chunks and embeddings.
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    summary = get_chat_index_summary(session_id)

    return {
        "chunks_created": len(documents),
        "total_chat_chunks": summary["chunks"],
        "total_chat_files": summary["files"]
    }

# ==========================================
# RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve(question, session_id="default", top_k=4):
    records = get_chat_records(session_id)

    if not records["documents"]:
        return []

    question_embedding = embed_texts([question])[0]

    # Retrieve only chunks from the current chat.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, len(records["ids"])),
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"]
    )

    result_documents = results.get("documents") or [[]]
    result_metadatas = results.get("metadatas") or [[]]
    result_distances = results.get("distances") or [[]]

    passages = []

    for document, metadata, distance in zip(
        result_documents[0],
        result_metadatas[0],
        result_distances[0]
    ):
        passages.append({
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", 1),
            "similarity": 1 - distance
        })

    return passages

# ==========================================
# GENERATE ANSWER
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
                        "Do not invent facts. If the answer is not in the "
                        "context, clearly say that it was not found. "
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
            f"Could not generate an answer. Check that Ollama "
            f"is running and '{LLM_MODEL}' is installed. "
            f"Details: {error}"
        ) from error

    return answer, passages