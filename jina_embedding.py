import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import shutil

class JinaEmbedding:
    def __init__(self, model_path="jina-embeddings-v3"):
        self.model_path = model_path
        
        # Verify PyTorch version
        if not hasattr(torch, 'compiler'):
            raise RuntimeError(
                "Jina embeddings require PyTorch 2.2+ with compiler support.\n"
                "Please upgrade with: pip install --upgrade torch torchvision torchaudio"
            )

        # First-time setup: download if directory doesn't exist or is incomplete
        if not os.path.exists(os.path.join(self.model_path, "config.json")):
            os.makedirs(self.model_path, exist_ok=True)
            print("Downloading Jina model...")
            try:
                model = AutoModel.from_pretrained(
                    "jinaai/jina-embeddings-v3",
                    trust_remote_code=True
                )
                model.save_pretrained(self.model_path)
                tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
                tokenizer.save_pretrained(self.model_path)
            except Exception as e:
                shutil.rmtree(self.model_path)
                raise RuntimeError(f"Model download failed: {e}")

    def jina(self):
        if hasattr(torch.backends, "disable_math_sdp"):
            torch.backends.disable_math_sdp = True
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        embedding_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model.to(device)
        return tokenizer, embedding_model, device
    
