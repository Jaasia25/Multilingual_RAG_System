import torch
from jina_embedding import JinaEmbedding  # import your embedding class
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel

class QueryChroma:
    def __init__(self, model_path="jina-embeddings-v3", db_path="chroma_db"):
        self.jina_model = JinaEmbedding(model_path)
        self.tokenizer, self.model, self.device = self.jina_model.jina()

        # Setup Chroma client
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=db_path,
            chroma_db_impl="duckdb+parquet"
        ))
        self.collection = self.chroma_client.get_collection(name="bangla_docs")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[:, 0, :]  # CLS token
            return embeddings[0].cpu().numpy().tolist()

    def query(self, user_query, top_k=3):
        embedding = self.get_embedding(user_query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        print("🔍 Top Results:")
        for i, doc in enumerate(results["documents"][0]):
            print(f"\n🔹 Result {i+1}:\n{doc}")


# Example usage
if __name__ == "__main__":
    qc = QueryChroma()
    query = input("❓ Enter your query: ")
    qc.query(query)




