# 📚 Enterprise Knowledge Copilot

An AI-powered document question-answering application built with **Python, Streamlit, Gemini API, ChromaDB, and Retrieval-Augmented Generation (RAG)**.

The application allows users to upload PDF or TXT documents, index their content, and ask questions. Relevant document chunks are retrieved using semantic search and provided to a Gemini language model to generate context-aware answers with source references.

## 🚀 Live Demo

**Streamlit App:**  
https://enterprise-knowledge-copilot-hgtgsr9kh2wlpn9ddj64pz.streamlit.app/

**GitHub Repository:**  
https://github.com/sushmitakundarnad/enterprise-knowledge-copilot

---

## ✨ Features

- 📄 Upload PDF and TXT documents
- 🔍 Extract text from uploaded documents
- ✂️ Split documents into smaller text chunks
- 🧠 Generate semantic embeddings using Gemini
- 🗄️ Store document embeddings in ChromaDB
- 🔎 Retrieve relevant document chunks using vector similarity search
- 🤖 Generate answers using Gemini
- 📚 Display source document names and page numbers
- 💬 Interactive chat-based interface
- 🔄 New Chat functionality for starting a fresh document session
- ☁️ Designed for cloud deployment using Gemini API

---

## 🛠️ Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Core application development |
| Streamlit | Web application and user interface |
| Gemini API | Embeddings and answer generation |
| `gemini-embedding-001` | Document and query embeddings |
| Gemini LLM | Context-based answer generation |
| ChromaDB | Vector database |
| PyMuPDF | PDF text extraction |
| RAG | Retrieval-Augmented Generation |
| Git \& GitHub | Version control and source management |

---

# 🧠 How the Project Works

The project follows a **Retrieval-Augmented Generation (RAG)** architecture.

``` text
                 USER
                   │
                   ▼
          ┌─────────────────┐
          │ Streamlit UI    │
          └────────┬────────┘
                   │
             Upload PDF/TXT
                   │
                   ▼
          ┌─────────────────┐
          │ Text Extraction │
          │    PyMuPDF      │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Text Chunking   │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Gemini Embedding│
          │     Model       │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │    ChromaDB     │
          │  Vector Store   │
          └─────────────────┘


             USER QUESTION
                   │
                   ▼
          ┌─────────────────┐
          │ Gemini Query    │
          │   Embedding     │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ ChromaDB Search │
          └────────┬────────┘
                   │
          Relevant Chunks
                   │
                   ▼
          ┌─────────────────┐
          │ Context + Query │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Gemini LLM    │
          └────────┬────────┘
                   │
                   ▼
              FINAL ANSWER
                   +
             SOURCE REFERENCES
```

---

# 🔄 RAG Workflow

## 1\. Document Upload

The user uploads a PDF or TXT document through the Streamlit interface.

## 2\. Text Extraction

For PDF files, **PyMuPDF** extracts the text page by page.

For TXT files, the application reads the text directly.

## 3\. Text Chunking

Large documents are divided into smaller chunks.

The current implementation uses:

``` text
Chunk size: 1000 characters
Overlap: 150 characters
```

The overlap helps preserve context between neighboring chunks.

## 4\. Generate Embeddings

Each document chunk is converted into a numerical vector using:

``` text
gemini-embedding-001
```

These embeddings represent the semantic meaning of the text.

## 5\. Store in ChromaDB

The generated embeddings, document text, and metadata are stored in **ChromaDB**.

Metadata includes:

- Source filename
- Page number
- Chat/session ID

## 6\. User Asks a Question

The user's question is converted into an embedding using the Gemini embedding model.

## 7\. Retrieve Relevant Information

ChromaDB compares the question embedding with the stored document embeddings and retrieves the most relevant chunks.

The application retrieves the top relevant passages for the question.

## 8\. Generate the Answer

The retrieved document context and user's question are sent to Gemini.

The prompt instructs the model to:

- Use only the supplied document context
- Avoid inventing information
- Clearly state when the answer is not found
- Use simple English

## 9\. Display Sources

The application displays the source filename and page numbers associated with the retrieved passages.

---

# 📂 Project Structure

``` text
enterprise-knowledge-copilot/
│
├── app.py
│
├── rag.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── chroma_db/
```

### `app.py`

Handles the Streamlit user interface, including:

- Page configuration
- Document upload
- Chunk preview
- Index Documents button
- Chat interface
- Chat history
- New Chat functionality
- Source references

### `rag.py`

Contains the core RAG functionality:

- PDF/TXT text extraction
- Text chunking
- Gemini embeddings
- ChromaDB storage
- Similarity search
- Gemini answer generation

### `requirements.txt`

Contains the Python dependencies required to run the project.

---

# ⚙️ Installation

## 1\. Clone the repository

``` bash
git clone https://github.com/sushmitakundarnad/enterprise-knowledge-copilot.git
```

Move into the project directory:

``` bash
cd enterprise-knowledge-copilot
```

## 2\. Create a virtual environment

### Windows

``` bash
python -m venv .venv
```

Activate it:

``` bash
.venv\Scripts\activate
```

### macOS/Linux

``` bash
python3 -m venv .venv
```

``` bash
source .venv/bin/activate
```

## 3\. Install dependencies

``` bash
pip install -r requirements.txt
```

---

# 🔑 Gemini API Configuration

This project uses the **Gemini API** instead of relying on a locally running Ollama server.

Create a Gemini API key through Google AI Studio.

Do not hard-code the API key inside the Python source code.

For local development, create:

``` text
.streamlit/
└── secrets.toml
```

Add:

``` toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

For Streamlit Cloud deployment, add the same secret through the application's **Secrets** settings.

### ⚠️ Security

Never commit your API key to GitHub.

Add your secrets file to `.gitignore`:

``` text
.streamlit/secrets.toml
```

---

# ▶️ Run the Application

Start the application with:

``` bash
streamlit run app.py
```

The application will open in your browser.

---

# 💬 Example Usage

### Step 1

Upload a PDF document.

### Step 2

Click:

``` text
Index Documents
```

### Step 3

Wait for the document to be indexed.

Example:

``` text
Document indexed successfully.
Chunks created: 3
```

### Step 4

Ask a question:

``` text
What is this document about?
```

### Step 5

The application:

``` text
Question
   ↓
Question Embedding
   ↓
ChromaDB Similarity Search
   ↓
Relevant Document Chunks
   ↓
Gemini
   ↓
Answer + Sources
```

---

# 🎯 Why RAG?

A normal LLM does not automatically know the contents of a private document uploaded by the user.

RAG allows the application to:

1. Retrieve relevant information from the user's documents.
2. Add that information to the model's context.
3. Generate an answer based on the retrieved information.

This makes the application useful for document-based knowledge systems.

---

# 🧩 Key Concepts Demonstrated

This project demonstrates practical understanding of:

- Generative AI
- Large Language Models
- Retrieval-Augmented Generation
- Text embeddings
- Semantic search
- Vector databases
- Prompt engineering
- Document processing
- Context retrieval
- AI application development
- Streamlit deployment
- API integration

---

# 📌 Limitations

This project is designed as a portfolio and learning application.

Current limitations include:

- Supports PDF and TXT files.
- Scanned/image-only PDFs may require OCR.
- ChromaDB is currently used as a local vector store.
- Gemini API usage is subject to API rate limits and quotas.
- The application is not designed as a production enterprise database.
- Authentication and role-based access control are not currently implemented.
- Large-scale document processing would require additional optimization.

---

# 🔮 Future Enhancements

Possible future improvements include:

- OCR support for scanned PDFs
- DOCX and additional document formats
- Persistent cloud vector database
- User authentication
- Role-based document access
- Improved retrieval evaluation
- Similarity score thresholds
- Conversation memory
- Streaming AI responses
- Document management dashboard
- Production monitoring and logging

---

# 📚 What I Learned

Through this project, I gained practical experience in building an end-to-end Generative AI application.

Key learning areas include:

- How RAG works
- How document chunking affects retrieval
- How embeddings represent text semantically
- How vector databases support similarity search
- How to connect an LLM to retrieved context
- How to build AI applications using Streamlit
- How to integrate cloud AI APIs
- How to deploy a Generative AI application

---

# 👩‍💻 Author

**Sushmita Kundarnad**

Computer Science Graduate | Generative AI | Python | AI Applications

GitHub:  
https://github.com/sushmitakundarnad

---

# ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
