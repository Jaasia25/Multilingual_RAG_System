import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb

class EmbeddingModel:
    def __init__(self, model_path="jina-embeddings-v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, task="text-matching"):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state[:, 0, :]
        return output.cpu().numpy()

class Vector_Bangla:
    def __init__(self, tokenizer, embedding_model, device, collection_name="rag_db", persist_dir="chroma_store"):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=persist_dir)

        if collection_name in [c.name for c in self.client.list_collections()]:
            print(f"✅ Using existing collection: '{collection_name}'")
            self.collection = self.client.get_collection(name=collection_name)
        else:
            print(f"📁 Creating new collection: '{collection_name}'")
            self.collection = self.client.create_collection(name=collection_name)

    def encode_bangla_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            sentences = file.readlines()
        sentences = [s.strip() for s in sentences if s.strip()]
        embeddings = self.embedding_model.encode(sentences, task="text-matching")

        for sentence, emb in zip(sentences, embeddings):
            metadata = {"source": str(file_path)}
            self.collection.add(
                documents=[sentence],
                embeddings=[emb.tolist()],
                metadatas=[metadata],
                ids=[str(hash(sentence))]
            )
        print(f"✅ Embedded and stored sentences from {file_path}")

    def encode_directory(self, directory_path):
        directory = Path(directory_path)
        for file_path in directory.glob("*.txt"):
            self.encode_bangla_text_file(file_path)

    def get_collection(self):
        return self.collection

def main():
    embedding_model = EmbeddingModel()
    vector_store = Vector_Bangla(
        tokenizer=embedding_model.tokenizer,
        embedding_model=embedding_model,
        device=embedding_model.device,
        collection_name="rag_db",
        persist_dir="chroma_store"
    )
    
    # Check if collection is empty (no documents)
    collection = vector_store.get_collection()
    try:
        count = len(collection.get(include=["ids"])["ids"])
    except Exception:
        count = 0

    if count == 0:
        print("Collection is empty. Starting to embed text files from 'input' folder...")
        vector_store.encode_directory("input")
    else:
        print(f"Collection already has {count} documents. Skipping embedding.")

if __name__ == "__main__":
    main()
