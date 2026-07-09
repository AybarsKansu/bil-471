import re
import pandas as pd
from datasets import load_from_disk
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# SADECE İKİ NOKTA VEYA ALT SATIR İLE BİTEN, KESİN BAŞLIKLAR
STRICT_PATTERNS = [
    re.compile(r'\b(?:H\s*Ü\s*K\s*Ü\s*M|S\s*O\s*N\s*U\s*Ç)\s*(?:[:：]|[\n\r]+)', re.IGNORECASE),
    re.compile(r'\bG\s*E\s*R\s*E\s*Ğ\s*İ\s+D\s*Ü\s*Ş\s*Ü\s*N\s*Ü\s*L\s*D\s*Ü\s*(?:[:：]|[\n\r]+)', re.IGNORECASE),
]

def extract_old(text):
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
            if len(verdict_text) > 25: 
                return verdict_text
    return None

def extract_new(text):
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

if __name__ == "__main__":
    print("Veri seti yükleniyor...")
    ds = load_from_disk(str(REPO_ROOT / "saved_dataset"))
    df = ds['train'].to_pandas()
    
    print("Orijinal (sınırsız) regex kuralı uygulanıyor...")
    df['old_output'] = df['text'].apply(extract_old)
    
    print("Yeni (1500 karakter sınırına sahip) regex kuralı uygulanıyor...")
    df['new_output'] = df['text'].apply(extract_new)
    
    # Sadece eskiden çıkarılıp, yenisinde elenenler
    eliminated_df = df[(df['old_output'].notna()) & (df['new_output'].isna())].copy()
    
    print(f"\nToplam eskiden çıkarılan kayıt sayısı: {df['old_output'].notna().sum()}")
    print(f"Toplam yeni kuralda çıkarılan kayıt sayısı: {df['new_output'].notna().sum()}")
    print(f"Elenen kayıt sayısı (>1500 karakter olanlar): {len(eliminated_df)}")
    
    if len(eliminated_df) > 0:
        eliminated_df['eliminated_length'] = eliminated_df['old_output'].apply(len)
        print(f"\n--- ELENEN KAYITLARIN İSTATİSTİKLERİ ---")
        print(f"Ortalama Uzunluk: {eliminated_df['eliminated_length'].mean():.2f} karakter")
        print(f"Minimum Uzunluk: {eliminated_df['eliminated_length'].min()} karakter")
        print(f"Maksimum Uzunluk: {eliminated_df['eliminated_length'].max()} karakter")
        
        # Analiz için 20 rastgele örneği bir txt dosyasına yazdır
        sample_size = min(20, len(eliminated_df))
        sample_df = eliminated_df.sample(n=sample_size, random_state=42)
        
        output_file = "eliminated_samples_analysis.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"1500 KARAKTER SINIRI YÜZÜNDEN ELENEN {sample_size} RASTGELE ÖRNEK ANALİZİ\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in sample_df.iterrows():
                f.write(f"KAYIT ID/INDEX: {idx} | UZUNLUK: {row['eliminated_length']} karakter\n")
                f.write("-" * 80 + "\n")
                f.write(f"ELENEN ÇIKTI (REGEX'İN BULDUĞU AMA ÇOK UZUN OLAN KISIM):\n")
                # Baştan ve sondan biraz gösterip çok uzunsa ortasını kırpabiliriz veya tamamını yazabiliriz.
                # Tamamını yazdıralım ki ne kadar saçma olduğu anlaşılsın.
                f.write(row['old_output'] + "\n")
                f.write("="*80 + "\n\n")
                
        print(f"\nElenen kayıtlardan {sample_size} tanesi rastgele seçilip detaylı analiz için '{output_file}' dosyasına kaydedildi.")
