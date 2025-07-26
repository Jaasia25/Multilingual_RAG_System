import os
import io
import re
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client as OllamaClient


# =========================
# Embedding + ChromaDB Storage (Vector_Bangla style)
# =========================
class Vector_Bangla:
    def __init__(self, tokenizer, embedding_model, device, collection_name="rag_db"):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device
        self.collection_name = collection_name
        self.client = chromadb.Client()

        # If collection exists, delete and recreate
        if collection_name in [c.name for c in self.client.list_collections()]:
            print(f"Collection '{collection_name}' exists. Deleting it...")
            self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(collection_name)

    def encode_bangla_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            sentences = file.readlines()
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Use embedding model's encode method (batch)
        embeddings = self.embedding_model.encode(sentences, task="text-matching")

        for sentence, emb in zip(sentences, embeddings):
            metadata = {"source": str(file_path)}
            self.collection.add(
                documents=[sentence],
                embeddings=[emb.tolist() if hasattr(emb, "tolist") else emb],
                metadatas=[metadata],
                ids=[str(hash(sentence))]
            )
        print(f"✅ Embedded and stored sentences from {file_path}")

    def encode_directory(self, directory_path):
        directory = Path(directory_path)
        for file_path in directory.glob("*.txt"):
            self.encode_bangla_text_file(file_path)

    def query(self, query_text, top_k=3):
        emb = self.embedding_model.encode([query_text], task="text-matching")[0]
        results = self.collection.query(query_embeddings=[emb.tolist() if hasattr(emb, "tolist") else emb], n_results=top_k)
        # returns list of docs in results['documents'][0]
        return results["documents"][0]

# =========================
# Embedding Model Wrapper (using HuggingFace transformer + PyTorch)
# for compatibility with your previous JinaEmbedding usage
# but providing encode() interface like sentence-transformers
# =========================
class EmbeddingModel:
    def __init__(self, model_path="jina-embeddings-v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, task="text-matching"):
        # texts: list of strings
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state[:, 0, :]
        return output.cpu().numpy()

# =========================
# Chatbot with LLaMA + retrieval
# =========================
class ChatWithLLaMA:
    def __init__(self, embedder, collection_name="rag_db", llama_model="llama3"):
        self.embedder = embedder
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            raise RuntimeError(f"❌ Collection '{collection_name}' not found. Please run embedding first.")
        self.llama = OllamaClient(host="http://localhost:11434")
        self.llama_model = llama_model

    def retrieve_context(self, query, top_k=3):
        emb = self.embedder.encode([query], task="text-matching")[0]
        results = self.collection.query(query_embeddings=[emb.tolist() if hasattr(emb, "tolist") else emb], n_results=top_k)
        return results["documents"][0]

    def generate_response(self, query, context_chunks, bangla=True):
        context = "\n\n".join(context_chunks)
        prompt = (
            f"{'তোমার কাছে নিম্নলিখিত তথ্য আছে' if bangla else 'You have the following information'}:\n\n"
            f"{context}\n\n"
            f"{'উপরের তথ্য ব্যবহার করে এই প্রশ্নের উত্তর দাও' if bangla else 'Using the above information, answer this question'}:\n{query}"
            f"{'বাংলায় উত্তর দাও' if bangla else 'Answer in English'}"
        )
        response = self.llama.chat(
            model=self.llama_model,
            messages=[
                {"role": "system", "content": "তুমি একজন সহায়ক বাংলা সহকারী।" if bangla else "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

# =========================
# FastAPI Setup
# =========================
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    bangla: bool = True

# Instantiate embedding model and vector store (global)
embedding_model = EmbeddingModel(model_path="jina-embeddings-v3")
vector_store = Vector_Bangla(
    tokenizer=embedding_model.tokenizer,
    embedding_model=embedding_model,
    device=embedding_model.device,
    collection_name="rag_db"
)

# Embed files before serving
vector_store.encode_directory("input")  # Embed all .txt files in the current folder

# Now instantiate the chatbot
chatbot = ChatWithLLaMA(embedder=embedding_model, collection_name="rag_db")

@app.get("/")
async def read_root():
    return {"message": "Welcome! Use POST /chat with JSON {'query': 'your question', 'bangla': true} to interact with the bot."}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        context = chatbot.retrieve_context(request.query)
        print(context)
        response = chatbot.generate_response(request.query, context, request.bangla)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
