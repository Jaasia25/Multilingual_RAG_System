<<<<<<< HEAD
# 🌐 Multilingual RAG System

A Retrieval-Augmented Generation (RAG) system that supports both Bangla and English languages for document-based question answering.

---

## 📦 Setup Guide

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jaasia25/Multilingual_RAG_System.git
   cd Multilingual_RAG_System


### ✅ Create and activate virtual environment (optional but recommended)

```bash
python -m venv venv
.env\Scriptsctivate  # On Windows
```

---

### ✅ Install dependencies

```bash
pip install -r requirements.txt
```

---

### ✅ Run the FastAPI app

```bash
uvicorn main:app --reload
```

---

# 🧰 Tools, Libraries, and Packages Used

- **Python 3.10+**
- **FastAPI** – Web framework for API development
- **Jina Embeddings** – Multilingual text embedding model
- **ChromaDB** – Vector database for document retrieval
- **Ollama / LLaMA3** – LLM backend for generating responses
- **PyMuPDF / OCR** – For PDF and image processing
- **Langdetect** – For auto-detecting Bangla/English
- **Streamlit** – (Optional) For frontend UI
- **Uvicorn** – ASGI server

---

# 💬 Sample Queries and Outputs

### ➤ English Example

**Query:**

```text
What is the interest rate for savings accounts?
```

**Response:**

```text
The current interest rate for savings accounts is 4.5% annually.
```

---

### ➤ Bangla Example

**Query (in Bangla):**

```text
সেভিংস অ্যাকাউন্টের ইন্টারেস্ট রেট কত?
```

**Response:**

```text
সেভিংস অ্যাকাউন্টের জন্য বর্তমান সুদের হার বার্ষিক ৪.৫%।
```

---

# 🔌 API Documentation

Once the FastAPI server is running, access the API docs at:

- **Swagger UI:** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc

---

## 📍 Available Endpoints

### 1. Upload PDF and Store Embeddings

```http
POST /upload
```

**Form Data:**

- `file`: PDF file (English or Bangla)  
- `bangla`: `true` or `false` (whether the document is in Bangla)

**Response:**

```json
{
  "message": "Embeddings stored successfully."
}
```

---

### 2. Query the Chatbot

```http
POST /query
```

**JSON Body:**

```json
{
  "query": "আপনার প্রশ্ন এখানে লিখুন",
  "bangla": true
}
```

**Response:**

```json
{
  "answer": "এখানে উত্তর আসবে"
}
```
=======

>>>>>>> dbab176884bc5642ab204554f8ccb13fc4786348
