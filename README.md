# 📄 Document Q&A with Groq

This is a **Streamlit-based web application** that allows users to **upload documents (PDF, TXT, and more)** and **interactively ask questions** based on their contents using **Groq LLM** and **LangChain**.

---

## 🚀 Features

- ✅ Upload and process `.pdf`, `.txt`, and unstructured files
- ✅ Ask questions about document contents
- ✅ Uses **LangChain**, **Groq (LLM)**, and **FAISS** for fast vector-based retrieval
- ✅ Embeddings powered by HuggingFace (`all-MiniLM-L6-v2`)
- ✅ Interactive and intuitive UI with Streamlit

---

## 🛠️ Tech Stack

| Tool              | Purpose                                  |
|------------------|------------------------------------------|
| `Streamlit`       | Frontend UI                              |
| `LangChain`       | LLM and retrieval chain orchestration    |
| `Groq (ChatGroq)` | LLM backend                              |
| `FAISS`           | Vector store for semantic search         |
| `HuggingFace`     | Text embeddings                          |
| `dotenv`          | Load API keys securely                   |

---

## 📂 File Upload Support

- PDF files (`.pdf`)
- Text files (`.txt`)
- Other unstructured files (fallback to generic loader)

