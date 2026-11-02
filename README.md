# 🧠 Enterprise Knowledge Copilot

An AI-powered document question-answering application that helps users search, understand, and retrieve information from their documents using **Retrieval-Augmented Generation (RAG)**.

The application allows users to upload PDF and TXT files, index their content, and ask questions in natural language. It retrieves relevant information from the uploaded documents and generates context-based answers using a locally running Large Language Model (LLM).

## 🚀 Features

* **📄 Document Upload:** Upload PDF and TXT documents.
* **📚 Document Indexing:** Extract document text, split it into smaller chunks, and store it in a vector database.
* **🔍 Semantic Search:** Retrieve relevant document content based on the meaning of a question.
* **🤖 AI-Powered Answers:** Generate answers using retrieved document context.
* **📌 Source References:** Display source filenames and page information when available.
* **💬 Interactive Chat:** Ask follow-up questions through a Streamlit interface.
* **📝 Suggested Questions:** Get suggested questions to explore document content.
* **🔒 Local LLM Inference:** Use Ollama to run the language model locally instead of requiring a paid hosted LLM API.
* **🗂️ Session-Based Documents:** Organize document retrieval by chat session when supported by the application configuration.

## 🛠️ Tech Stack

| Technology                           | Purpose                                      |
| ------------------------------------ | -------------------------------------------- |
| Python                               | Core application logic                       |
| Streamlit                            | Interactive web interface                    |
| Ollama                               | Run language and embedding models locally    |
| Llama 3.2 (3B)                       | Generate natural-language answers            |
| Nomic Embed Text                     | Convert text into vector embeddings          |
| ChromaDB                             | Store and retrieve document embeddings       |
| PyMuPDF                              | Extract text from PDF files                  |
| Retrieval-Augmented Generation (RAG) | Ground answers in retrieved document content |

## 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │       User Interface    │
                 │        Streamlit        │
                 └────────────┬────────────┘
                              │
                  Upload Documents / Ask
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Document Processing  │
                 │      PDF / TXT          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Text Extraction     │
                 │       PyMuPDF           │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Text Chunking       │
                 │ Split into small pieces │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Embedding Model      │
                 │    Nomic Embed Text     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     ChromaDB            │
                 │   Vector Database       │
                 └────────────┬────────────┘
                              │
                       User Question
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Semantic Retrieval    │
                 │ Find Relevant Chunks    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Llama 3.2 (3B)     │
                 │ Generate Context-Based  │
                 │         Answer          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Answer + Source Details│
                 │ Displayed in Streamlit  │
                 └─────────────────────────┘
```

## 🔄 How It Works

### 1. Upload Documents

The user uploads PDF or TXT documents through the Streamlit interface.

### 2. Extract Text

The application extracts text from the uploaded files. PyMuPDF is used to process PDF documents.

### 3. Split Text into Chunks

The extracted text is divided into smaller sections so that relevant information can be retrieved efficiently.

### 4. Generate Embeddings

The Nomic Embed Text model converts text chunks into numerical vector representations that capture semantic meaning.

### 5. Store Embeddings

ChromaDB stores the embeddings alongside document text and metadata, such as filenames and page numbers.

### 6. Retrieve Relevant Information

When the user asks a question, the application searches for document chunks that are semantically similar to the question.

### 7. Generate an Answer

The retrieved content is supplied to Llama 3.2 (3B) through Ollama. The model uses this context to generate a natural-language response.

### 8. Display Results

The answer is displayed in the Streamlit interface, along with source information when available.

## 💻 Installation and Setup

### Prerequisites

Install the following before running the project:

* Python 3.10 or a compatible version
* Git
* Ollama

### Step 1: Clone the Repository

```bash
git clone https://github.com/sushmitakundarnad/enterprise-knowledge-copilot.git
cd enterprise-knowledge-copilot
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Ollama Models

Make sure Ollama is installed and running. Then open a terminal and run:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

These commands download the models required by the application.

### Step 5: Start the Application

Run the following command from the project folder:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 📁 Project Structure

```text
Enterprise-Knowledge-Copilot/
│
├── app.py              # Streamlit user interface and application flow
├── rag.py              # Document processing, embeddings, retrieval, and RAG logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Files excluded from Git
└── chroma_db/           # Local vector database, if configured
```

The `chroma_db/` directory may be created automatically when the application initializes its persistent vector store.

## 🧠 What Is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with generative AI.

Instead of asking an AI model to answer a question using only its pre-trained knowledge, RAG retrieves relevant information from a supplied knowledge source and uses that information to generate an answer.

In this project:

* **Retrieval:** ChromaDB helps find relevant document chunks.
* **Augmentation:** The retrieved chunks provide context for the question.
* **Generation:** Llama 3.2 generates a natural-language answer using the provided context.

This approach can help make answers more relevant to the uploaded documents, although the model can still make mistakes.

## 🎯 Use Cases

* Searching internal company documentation
* Understanding policies and procedures
* Finding information in reports and manuals
* Exploring technical documentation
* Asking questions about PDF-based knowledge resources
* Building a foundation for enterprise knowledge assistants

## 🔮 Future Improvements

* Support DOCX and additional document formats.
* Add OCR for scanned PDFs.
* Improve source citations with page-level references and text excerpts.
* Add user authentication and access controls.
* Support persistent chat history.
* Add document management and deletion.
* Improve retrieval using reranking and hybrid search.
* Add evaluation tools to measure retrieval and answer quality.
* Deploy a public demo using a compatible hosted inference solution.

## ⚠️ Deployment Notes

This project uses Ollama for local model inference. **A standard Streamlit Community Cloud deployment cannot automatically access Ollama running on your personal computer.**

To deploy the application publicly, configure an inference service that the deployed application can reach, or host the application and Ollama in an appropriately configured environment.

ChromaDB persistence on a hosted platform also depends on its storage configuration. Do not assume locally stored documents or indexes will survive restarts on an ephemeral deployment.

Do not upload API keys, passwords, private documents, or other sensitive information to GitHub.

## 👨‍💻 Author

**Sushmita Kundarnad**

Aspiring Generative AI Developer | Python | RAG | LLMs | Vector Databases

## 🌐 Live Demo

🚀 **Try the application here** : https://enterprise-knowledge-copilot-hgtgsr9kh2wlpn9ddj64pz.streamlit.app/

Upload your PDF or TXT documents, ask questions in natural language, and explore how Retrieval-Augmented Generation (RAG) helps retrieve relevant information from your documents.
