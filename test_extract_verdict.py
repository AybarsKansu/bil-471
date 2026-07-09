import re
import pandas as pd
from datasets import load_from_disk
import time
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# SADECE İKİ NOKTA VEYA ALT SATIR İLE BİTEN, KESİN BAŞLIKLAR
STRICT_PATTERNS = [
    re.compile(r'\b(?:H\s*Ü\s*K\s*Ü\s*M|S\s*O\s*N\s*U\s*Ç)\s*(?:[:：]|[\n\r]+)', re.IGNORECASE),
    re.compile(r'\bG\s*E\s*R\s*E\s*Ğ\s*İ\s+D\s*Ü\s*Ş\s*Ü\s*N\s*Ü\s*L\s*D\s*Ü\s*(?:[:：]|[\n\r]+)', re.IGNORECASE),
]

def extract_verdict_pure(text):
    normalized = text.replace('\xa0', ' ').replace('**', '')
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    
    text_length = len(normalized)
    
    for pattern in STRICT_PATTERNS:
        matches = list(pattern.finditer(normalized))
        
        # Başlık mutlaka metnin son %50'sinde olmalı
        valid_matches = [m for m in matches if m.start() > (text_length * 0.5)]
        
        if valid_matches:
            last_match = valid_matches[-1]
            verdict_text = normalized[last_match.end():].strip()
            
            # Hüküm metni 25 karakterden uzunsa
            if len(verdict_text) > 25: 
                return verdict_text

    # Başlık tam olarak yoksa veya 25 karakterden kısaysa None dön.
    return None


if __name__ == "__main__":
    print("Loading the dataset...")
    ds = load_from_disk(str(REPO_ROOT / "saved_dataset"))
    
    df = ds['train'].to_pandas()
    
    print("PERFORMING EXTRACTION ON THE ENTIRE DATASET USING STRICT LOGIC...")
    start_time = time.time()
    df['hukum_text'] = df['text'].apply(extract_verdict_pure)
    end_time = time.time()
    
    extracted_count = df['hukum_text'].notna().sum()
    total_count = len(df)
    rate = (extracted_count / total_count) * 100
    
    print(f"\nTotal records processed: {total_count}")
    print(f"Confidently extracted: {extracted_count}")
    print(f"Remaining records: {total_count - extracted_count}")
    print(f"Success rate: {rate:.2f}%")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    
    output_filename = "extracted_verdicts_output.txt"
    print(f"\nSaving the extracted texts to {output_filename}...")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" OUTPUT OF ALL EXTRACTED SAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for idx, row in df[df['hukum_text'].notna()].iterrows():
            f.write(f"DECISION INDEX: {idx}\n")
            f.write("-" * 40 + "\n")
            
            original_tail = row['text'][-300:].replace('\n', '  ')
            f.write(f"LATTER PORTION OF THE ORIGINAL TEXT:\n...{original_tail}\n\n")
            
            hukum = row['hukum_text']
            hukum_preview = hukum
            hukum_preview = hukum_preview.replace('\n', '  ')
            f.write(f"TEXT EXTRACTED BY REGEX:\n{hukum_preview}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
    print(f"Extraction results successfully saved to {output_filename}")
