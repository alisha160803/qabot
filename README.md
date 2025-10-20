# 📄 Document QA Bot

This is a Gradio-powered chatbot that answers questions based on uploaded PDF documents using IBM Watsonx and LangChain. It extracts text from the PDF, embeds it using Watsonx Embeddings, stores it in ChromaDB, and uses a RetrievalQA chain to generate answers using Watsonx LLM.

---

## 🚀 Features

- Upload any readable PDF
- Ask questions about its content
- Powered by IBM Watsonx LLM and Embeddings
- Uses LangChain + ChromaDB for retrieval
- Simple Gradio interface

---

## 🧠 How It Works

- **PDF Upload**: Users upload a readable, text-based PDF.
- **Text Extraction**: The document is split into chunks using LangChain's `RecursiveCharacterTextSplitter`.
- **Embedding**: Each chunk is embedded using IBM Watsonx Embeddings (`slate.125m.english`).
- **Vector Store**: Embeddings are stored in ChromaDB for retrieval.
- **LLM Response**: IBM Watsonx LLM (`granite-3-2-8b-instruct`) generates answers based on retrieved chunks.
- **Interface**: Gradio provides a simple UI for uploading PDFs and entering questions.

---

## 🧪 IBM Skills Network Lab Instructions

If you're running this inside the IBM Skills Network lab environment:

1. Open the terminal and run:
   ```bash
   python qabot.py
