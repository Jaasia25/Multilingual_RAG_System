import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import re
import os
from tqdm import tqdm

class ExtractData:
    def __init__(self, tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        os.makedirs("input", exist_ok=True)

    def extract_text_from_pdf(self, pdf_path, lang="eng+ben"):
        doc = fitz.open(pdf_path)
        all_text = ""

        print(f"📄 Extracting text from {len(doc)} pages...")
        for i, page in enumerate(tqdm(doc, desc="🔍 Processing pages", unit="page")):
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img, lang=lang)
            all_text += f"\n{text}\n"

        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(all_text)

        print("✅ Raw text saved to output.txt")

    def pre_processing_and_chunk(self, input_file="output.txt", output_folder="input", min_sentences=20):
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Clean text
        cleaned = re.sub(r"[a-zA-Z0-9]", "", text)
        cleaned = re.sub(r"[()\[\]{}<>@#&*%$^+=|~`]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Split into sentences using "।"
        sentences = [s.strip() + "।" for s in cleaned.split("।") if s.strip()]

        # Group into chunks of at least `min_sentences`
        chunk = []
        chunk_count = 0
        for i, sentence in enumerate(sentences):
            chunk.append(sentence)
            if len(chunk) >= min_sentences:
                chunk_text = " ".join(chunk).strip()
                with open(os.path.join(output_folder, f"chunk_{chunk_count}.txt"), "w", encoding="utf-8") as f:
                    f.write(chunk_text)
                chunk_count += 1
                chunk = []

        # Save the last remaining chunk (if any)
        if chunk:
            chunk_text = " ".join(chunk).strip()
            with open(os.path.join(output_folder, f"chunk_{chunk_count}.txt"), "w", encoding="utf-8") as f:
                f.write(chunk_text)

        print(f"✅ Saved {chunk_count + 1} chunk(s) to '{output_folder}'")

# Example usage
if __name__ == "__main__":
    extractor = ExtractData()
    extractor.extract_text_from_pdf("HSC26-Bangla1st-Paper.pdf")
    extractor.pre_processing_and_chunk()
