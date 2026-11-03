# 📚 Enterprise Knowledge Copilot

An AI-powered document assistant that allows users to upload PDF and TXT files and ask questions about their content. It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document text and generate answers using a locally running Large Language Model (LLM) through Ollama.

## 🚀 Live Demo

**[Try Enterprise Knowledge Copilot](https://enterprise-knowledge-copilot-hgtgsr9kh2wlpn9ddj64pz.streamlit.app/)**

> Note: The deployed interface may be available even when document indexing or question answering is unavailable. The current implementation uses Ollama, which must be accessible to the running application.

## ✨ Features

* **Document Upload:** Upload PDF and TXT files.
* **Text Extraction:** Extract readable text from uploaded documents.
* **Text Chunking:** Split document content into smaller chunks for retrieval.
* **Document Indexing:** Generate embeddings and store document chunks in ChromaDB.
* **AI Question Answering:** Ask questions about indexed documents using RAG.
* **Semantic Retrieval:** Retrieve relevant text based on the user's question.
* **Source References:** Display source filenames and page numbers with answers.
* **Interactive UI:** Use a Streamlit interface with a midnight-blue theme.
* **Local LLM Support:** Generate responses using Ollama.

## 🛠️ Tech Stack

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| Python           | Core application logic                       |
| Streamlit        | User interface                               |
| Ollama           | Local LLM inference and embedding generation |
| Llama 3.2 (3B)   | Answer generation                            |
| Nomic Embed Text | Text embeddings                              |
| ChromaDB         | Vector storage and retrieval                 |
| PyMuPDF          | PDF text extraction                          |
| RAG              | Document-grounded question answering         |

## ⚙️ How It Works

1. Upload a PDF or TXT document.
2. The application extracts the document's text.
3. The text is divided into smaller chunks.
4. Ollama generates embeddings for the chunks.
5. ChromaDB stores the chunks and their embeddings.
6. Enter a question about your document.
7. The application retrieves relevant chunks using semantic search.
8. The LLM generates an answer using the retrieved context.
9. Source filenames and page references are displayed where available.

## 🧠 RAG Architecture

```text
PDF / TXT Upload
       |
       v
 Text Extraction
       |
       v
  Text Chunking
       |
       v
 Embedding Model
 (Nomic Embed Text)
       |
       v
    ChromaDB
       |
       v
  User Question
       |
       v
 Question Embedding
       |
       v
 Semantic Retrieval
       |
       v
 Relevant Text Chunks
       |
       v
  Ollama LLM
 (Llama 3.2 3B)
       |
       v
 Answer + Sources
```

## 📂 Project Structure

```text
enterprise-knowledge-copilot/
│
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/        # Local vector database, if created
```

The local `chroma_db/` directory should generally be excluded from Git because it contains generated application data.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sushmitakundarnad/enterprise-knowledge-copilot.git
cd enterprise-knowledge-copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If required, your `requirements.txt` should include:

```text
streamlit
chromadb
pymupdf
ollama
```

### 4. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Make sure the Ollama service is running on the computer that runs the application.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## 📖 How to Use

1. Open the application.
2. Expand **Upload Documents**.
3. Upload one or more PDF or TXT files.
4. Review the estimated text chunk count.
5. Click **Index Documents** and wait for the result.
6. Enter a question in the chat box.
7. Review the answer and expand **View Sources** to inspect source filenames and page references.

## 🔐 Privacy and Limitations

* With a local setup, model inference and embedding generation are handled by Ollama on the machine running the application.
* Uploaded document text is processed for retrieval and answer generation.
* Scanned PDFs without a readable text layer may require OCR.
* Answers depend on the quality of extracted text and retrieved chunks.
* The current version uses a local ChromaDB database.
* **Deployment consideration:** Streamlit Community Cloud cannot access Ollama running only on your personal computer. Cloud indexing and question answering require Ollama to be hosted somewhere the deployed app can reach, or an alternative hosted model service.

## 🔮 Future Improvements

* Support for DOCX documents.
* OCR for scanned PDFs.
* Conversation history and persistent user sessions.
* Improved retrieval and answer evaluation.
* More detailed source citations.
* Authentication and multi-user document management.

## 👩‍💻 Author

**Sushmita Kundarnad**

GitHub: [sushmitakundarnad](https://github.com/sushmitakundarnad)
