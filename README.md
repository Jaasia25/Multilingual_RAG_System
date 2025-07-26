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
.venv\Scripts\activate  # On Windows
```

---

### ✅ Install dependencies

```bash
pip install -r requirements.txt
```

---

### ✅ Extract the text through Pytesseract OCR

```bash
python extract_data.py
```
---

### ✅ Save the text to chroma (Vector database)

```bash
python vector_store.py
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
- **Pytesseract(Bn+En)** - For Bangla and English text extraction from PDF
- **PyMuPDF / OCR** – For PDF and image processing
- **Langdetect** – For auto-detecting Bangla/English
- **Uvicorn** – ASGI server

---

# 💬 Sample Queries and Outputs

### ➤ English Example

**Query:**

```text
Who is referred to as the god of luck in Anupam?
```

**Response:**

```text
According to the text, Lakshmi (the goddess of wealth and good fortune) is referred to as the god of luck for Anupam.
```

### FASTAPI English Response

![English Response](readme_images\10ms2.png)



---

### ➤ Bangla Example

**Query (in Bangla):**

```text
কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
```

**Response:**

```text
According to the text, মামাকে (Mama) is referred to as ভাগ্য দেবতা (Luck God) by Anupom's character. This means that Anupom considers his uncle (Mama) to be a person of great influence and importance in his life, much like a deity who can bring good or bad luck.
```

### FASTAPI Bangla Response

![Bangla Response](readme_images\10ms1.png)


---

# 🔌 API Documentation

Once the FastAPI server is running, access the API docs at:

- **Swagger UI:** http://localhost:8000/docs  

---

## 📍 Available Endpoints



---

### Query the Chatbot

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
  "response": "এখানে উত্তর আসবে"
}
```
=======
