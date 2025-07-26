import os
import glob
import time
import requests
from tqdm import tqdm

class BanglaChunkSummarizer:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"
        os.makedirs("summary", exist_ok=True)  # Ensure summary folder exists

    def summarize_bangla_chunk(self, text):
        prompt = (
            "তুমি একজন দক্ষ বাংলা ভাষার সহকারী। নিচের অনুচ্ছেদটি পড়ে সহজ ভাষায় মূল বক্তব্যটি বাংলায় তুলে ধরো:\n\n"
            f"{text.strip()}\n\n"
            "👉 সংক্ষেপে মূল বক্তব্য:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            return f"❌ অনুরোধ ব্যর্থ হয়েছে: {e}"
        except KeyError:
            return "❌ সার্ভার থেকে সঠিক উত্তর পাওয়া যায়নি।"

    def summarize_all_chunks(self, chunk_folder="input", summary_folder="summary"):
        chunk_files = sorted(glob.glob(os.path.join(chunk_folder, "chunk_*.txt")))
        if not chunk_files:
            print("⚠️ কোনো chunk_*.txt ফাইল পাওয়া যায়নি।")
            return

        for idx, file_path in enumerate(tqdm(chunk_files, desc="Summarizing Chunks")):
            with open(file_path, "r", encoding="utf-8") as f:
                chunk_text = f.read().strip()

            if not chunk_text:
                continue

            summary = self.summarize_bangla_chunk(chunk_text)

            summary_path = os.path.join(summary_folder, f"summary_{idx+1}.txt")
            with open(summary_path, "w", encoding="utf-8") as sf:
                sf.write(summary)

            time.sleep(1)  # Pause to prevent overload


# === Example usage ===
if __name__ == "__main__":
    summarizer = BanglaChunkSummarizer(model_name="llama3")
    summarizer.summarize_all_chunks()
