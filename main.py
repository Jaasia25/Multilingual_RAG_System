import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client as OllamaClient
import chromadb
from transformers import AutoTokenizer, AutoModel
from vector_store import EmbeddingModel, Vector_Bangla  # Import your vector_store.py classes

# =========================
# Chatbot with LLaMA + retrieval
# =========================
class ChatWithLLaMA:
    def __init__(self, embedder, collection_name="rag_db", persist_dir="chroma_store", llama_model="llama3"):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Loaded collection '{collection_name}' from persistent storage.")
        except Exception as e:
            raise RuntimeError(f"❌ Collection '{collection_name}' not found. Please run embedding first.") from e

        self.llama = OllamaClient(host="http://localhost:11434")
        self.llama_model = llama_model

    def retrieve_context(self, query, top_k=3):
        emb = self.embedder.encode([query], task="text-matching")[0]
        results = self.collection.query(
            query_embeddings=[emb.tolist() if hasattr(emb, "tolist") else emb], n_results=top_k
        )
        return results["documents"][0]

    def generate_response(self, query, context_chunks, bangla=True):
        context = "\n\n".join(context_chunks)
        prompt = (
            f"{'তোমার কাছে নিম্নলিখিত তথ্য আছে' if bangla else 'You have the following information'}:\n\n"
            f"{context}\n\n"
            f"{'উপরের তথ্য ব্যবহার করে এই প্রশ্নের উত্তর দাও' if bangla else 'Using the above information, answer this question'}:\n{query} "
            f"{'শুদ্ধু বাংলায় উত্তর দাও' if bangla else 'Answer in English'}"
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

# Instantiate embedding model and vector store (load persistent store only)
embedding_model = EmbeddingModel(model_path="jina-embeddings-v3")
vector_store = Vector_Bangla(
    tokenizer=embedding_model.tokenizer,
    embedding_model=embedding_model,
    device=embedding_model.device,
    collection_name="rag_db",
    persist_dir="chroma_store"
)

# Instantiate chatbot
chatbot = ChatWithLLaMA(embedder=embedding_model, collection_name="rag_db", persist_dir="chroma_store")

@app.get("/")
async def read_root():
    return {"message": "Welcome! Use POST /chat with JSON {'query': 'your question', 'bangla': true} to interact with the bot."}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        context = chatbot.retrieve_context(request.query)
        print("Retrieved context:", context)
        response = chatbot.generate_response(request.query, context, request.bangla)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
