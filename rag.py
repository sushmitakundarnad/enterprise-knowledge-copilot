import hashlib
import math
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
# EXTRACT TEXT FROM PDF AND TXT
# ==========================================

def extract_pages(filename, file_bytes):
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            pages = [
                (i + 1, page.get_text())
                for i, page in enumerate(pdf)
            ]

    elif filename_lower.endswith(".txt"):
        pages = [
            (1, file_bytes.decode("utf-8", errors="replace"))
        ]

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    return pages


# ==========================================
# SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text(text, chunk_size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    step = chunk_size - overlap

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), step)
        if text[i:i + chunk_size].strip()
    ]


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

def embed_texts(texts):
    if not texts:
        return []

    result = ollama.embed(
        model=EMBED_MODEL,
        input=texts
    )

    return result["embeddings"]


# ==========================================
# GET ALL STORED RECORDS
# ==========================================

def get_all_records():
    """
    Retrieve stored records without using ChromaDB filters.
    This avoids relying on filtered get() or count() calls.
    """

    records = collection.get(
        include=["documents", "metadatas"]
    )

    return records


# ==========================================
# COUNT DOCUMENT CHUNKS FOR A CHAT
# ==========================================

def get_chat_document_count(session_id):
    records = get_all_records()

    return sum(
        1
        for metadata in (records.get("metadatas") or [])
        if metadata and metadata.get("session_id") == session_id
    )


# ==========================================
# INDEX UPLOADED DOCUMENT
# ==========================================

def ingest_file(filename, file_bytes, session_id="default"):

    pages = extract_pages(filename, file_bytes)

    documents = []
    metadatas = []
    ids = []

    for page_number, text in pages:

        chunks = split_text(text)

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
            "No readable text found. "
            "If this is a scanned PDF, OCR may be required."
        )

    # Check Ollama before proceeding
    try:
        embeddings = []

        for start in range(0, len(documents), 16):
            batch = documents[start:start + 16]
            embeddings.extend(embed_texts(batch))

    except Exception as error:
        raise RuntimeError(
            f"Could not generate embeddings. "
            f"Check that Ollama is running and "
            f"'{EMBED_MODEL}' is installed. Details: {error}"
        ) from error

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "The number of embeddings does not match the chunks."
        )

    # Remove an older copy of this file in the same chat
    records = get_all_records()
    old_ids = []

    for record_id, metadata in zip(
        records.get("ids", []),
        records.get("metadatas") or []
    ):
        if (
            metadata
            and metadata.get("session_id") == session_id
            and metadata.get("source") == filename
        ):
            old_ids.append(record_id)

    if old_ids:
        collection.delete(ids=old_ids)

    # Save new chunks and their embeddings
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


# ==========================================
# CALCULATE COSINE SIMILARITY
# ==========================================

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==========================================
# RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve(question, session_id="default", top_k=4):

    records = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    embeddings = records.get("embeddings")

    if embeddings is None:
        raise RuntimeError(
            "Stored embeddings are unavailable. "
            "Please check your ChromaDB installation."
        )

    # Keep only documents belonging to this chat
    candidates = []

    for document, metadata, embedding in zip(
        documents, metadatas, embeddings
    ):
        if (
            metadata
            and metadata.get("session_id") == session_id
        ):
            candidates.append({
                "text": document,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", 1),
                "embedding": embedding
            })

    if not candidates:
        return []

    # Embed the question
    question_embedding = embed_texts([question])[0]

    # Rank the chunks by cosine similarity
    for item in candidates:
        item["similarity"] = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    # Return the most relevant chunks
    results = []

    for item in candidates[:top_k]:
        results.append({
            "text": item["text"],
            "source": item["source"],
            "page": item["page"],
            "similarity": item["similarity"]
        })

    return results


# ==========================================
# GENERATE ANSWER USING OLLAMA
# ==========================================

def answer_question(question, session_id="default"):

    passages = retrieve(
        question,
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
        f"Source: {item['source']}, "
        f"Page: {item['page']}\n"
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
                        "Answer questions using the supplied document "
                        "context only. Do not invent facts. "
                        "If the context does not contain the answer, "
                        "say that the information was not found. "
                        "Mention relevant source filenames and page "
                        "numbers when possible."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ]
        )

        answer = response["message"]["content"]

    except Exception as error:
        raise RuntimeError(
            f"Could not generate an answer. "
            f"Check that Ollama is running and "
            f"'{LLM_MODEL}' is installed. Details: {error}"
        ) from error

    return answer, passages


# ==========================================
# GENERATE SUGGESTED QUESTIONS
# ==========================================

def generate_suggested_questions(
    number=4,
    session_id="default"
):

    records = collection.get(
        include=["documents", "metadatas"]
    )

    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    # Select chunks belonging to the current chat
    chat_documents = [
        document
        for document, metadata in zip(documents, metadatas)
        if metadata and metadata.get("session_id") == session_id
    ]

    if not chat_documents:
        return []

    # Limit the context sent to the language model
    context = "\n\n".join(chat_documents[:8])

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Based on these document excerpts, generate "
                        f"{number} useful questions a reader could ask. "
                        "Return only questions, one per line.\n\n"
                        f"{context}"
                    )
                }
            ],
            options={"temperature": 0.3}
        )

        lines = response["message"]["content"].splitlines()

        questions = []

        for line in lines:
            question = re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line
            ).strip()

            if question.endswith("?") and question not in questions:
                questions.append(question)

        return questions[:number]

    except Exception:
        return []