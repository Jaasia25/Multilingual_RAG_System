import os
from jina_embedding import JinaEmbedding  # assuming your class is in jina_embedding.py
import torch
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel


class EmbedAndStore:
    def __init__(self, model_path="jina-embeddings-v3", db_path="chroma_db"):
        self.jina_model = JinaEmbedding(model_path)
        self.tokenizer, self.model, self.device = self.jina_model.jina()

        # Setup Chroma client
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=db_path,
            chroma_db_impl="duckdb+parquet"
        ))

        self.collection = self.chroma_client.get_or_create_collection(name="bangla_docs")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
            return embeddings[0].cpu().numpy().tolist()

    def embed_and_store(self, input_file="output.txt"):
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Split text into smaller chunks if needed
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]

        for idx, chunk in enumerate(chunks):
            if chunk.strip():  # skip empty strings
                embedding = self.get_embedding(chunk)
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"doc-{idx}"]
                )

        self.chroma_client.persist()
        print("✅ Embeddings stored successfully in Chroma DB.")


# Example usage
if __name__ == "__main__":
    embedder = EmbedAndStore()
    embedder.embed_and_store("output.txt")



