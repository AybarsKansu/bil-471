import re
import pandas as pd
from datasets import load_from_disk, Dataset
import os

STRICT_PATTERNS = [
    re.compile(r'\b(?:H\s*Ü\s*K\s*Ü\s*M|S\s*O\s*N\s*U\s*Ç)\s*(?:[:：]|[\n\r]+)', re.IGNORECASE),
    re.compile(r'\bG\s*E\s*R\s*E\s*Ğ\s*İ\s+D\s*Ü\s*Ş\s*Ü\s*N\s*Ü\s*L\s*D\s*Ü\s*(?:[:：]|[\n\r]+)', re.IGNORECASE),
]

def extract_verdict_pure_limited(text):
    normalized = text.replace('\xa0', ' ').replace('**', '')
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    text_length = len(normalized)
    
    for pattern in STRICT_PATTERNS:
        matches = list(pattern.finditer(normalized))
        valid_matches = [m for m in matches if m.start() > (text_length * 0.5)]
        if valid_matches:
            last_match = valid_matches[-1]
            verdict_text = normalized[last_match.end():].strip()
            if 25 < len(verdict_text) <= 1500: 
                return verdict_text
    return None

def main():
    print("Orijinal veri seti yükleniyor...")
    ds = load_from_disk(r'C:\work environment\Python\nlp_exercises\saved_dataset')
    df = ds['train'].to_pandas()
    
    print("Regex kuralı (1500 karakter sınırı) uygulanıyor...")
    df['hukum_text_regex'] = df['text'].apply(extract_verdict_pure_limited)
    
    # Sadece regex'in başarılı sonuç bulduğu kayıtları filtrele
    df_filtered = df[df['hukum_text_regex'].notna()].copy()
    print(f"Toplam geçerli kayıt sayısı: {len(df_filtered)}")
    
    # Test veri seti (Gold Dataset) için rastgele 15.000 kayıt seç
    test_size = 15000
    df_test = df_filtered.sample(n=test_size, random_state=42).copy()
    
    # Geriye kalanlar Eğitim veri seti (Train Dataset) olacak
    df_train = df_filtered.drop(df_test.index).copy()
    
    print(f"Test veri seti boyutu: {len(df_test)}")
    print(f"Eğitim veri seti boyutu: {len(df_train)}")
    
    # Yeniden Dataset nesnelerine dönüştür
    test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))
    train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
    
    # Diske kaydet
    test_path = "dataset_gold_test"
    train_path = "dataset_train"
    
    test_dataset.save_to_disk(test_path)
    train_dataset.save_to_disk(train_path)
    
    print(f"\nVeri setleri başarıyla kaydedildi:")
    print(f"- Test: {test_path}")
    print(f"- Train: {train_path}")

if __name__ == "__main__":
    main()
