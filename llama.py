import torch
from jina_embedding import JinaEmbedding
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from ollama import Client as OllamaClient

class ChatWithLLaMA:
    def __init__(self, model_path="jina-embeddings-v3", db_path="chroma_db", llama_model="llama3"):
        self.jina_model = JinaEmbedding(model_path)
        self.tokenizer, self.model, self.device = self.jina_model.jina()

        # Chroma vector DB setup
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=db_path,
            chroma_db_impl="duckdb+parquet"
        ))
        self.collection = self.chroma_client.get_collection(name="bangla_docs")

        # LLaMA client via Ollama
        self.llama = OllamaClient(host="http://localhost:11434")
        self.llama_model = llama_model

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[:, 0, :]
            return embeddings[0].cpu().numpy().tolist()

    def retrieve_context(self, query, top_k=3):
        embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results["documents"][0]

    def generate_response(self, query, context_chunks):
        context = "\n\n".join(context_chunks)
        prompt = (
            f"তোমার কাছে নিম্নলিখিত তথ্য আছে:\n\n"
            f"{context}\n\n"
            f"উপরের তথ্য ব্যবহার করে এই প্রশ্নের উত্তর দাও:\n{query}"
        )

        response = self.llama.chat(model=self.llama_model, messages=[
            {"role": "system", "content": "তুমি একজন সহায়ক বাংলা সহকারী।"},
            {"role": "user", "content": prompt}
        ])
        return response['message']['content']

    def chat(self):
        while True:
            query = input("\n❓ প্রশ্ন করুন (exit লিখে বন্ধ করুন): ")
            if query.lower() == "exit":
                break
            context_chunks = self.retrieve_context(query)
            response = self.generate_response(query, context_chunks)
            print(f"\n🤖 উত্তর:\n{response}")


# Example usage
if __name__ == "__main__":
    chatbot = ChatWithLLaMA()
    chatbot.chat()



