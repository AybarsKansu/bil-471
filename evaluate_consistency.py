import re
import os
import pandas as pd
from datasets import load_from_disk
import requests
from difflib import SequenceMatcher
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Regex Kalıpları
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
        valid_matches = [m for m in matches if m.start() > (text_length * 0.5)]
        if valid_matches:
            last_match = valid_matches[-1]
            verdict_text = normalized[last_match.end():].strip()
            if len(verdict_text) > 25: 
                return verdict_text
    return None

def extract_verdict_with_llm(text):
    prompt = f"Aşağıdaki hukuki metnin sadece hüküm (sonuç/karar) kısmını aynen yaz. Ekstra hiçbir yorum veya açıklama ekleme:\n\n{text[-2000:]}"
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0} # Extraction (çıkarım) işlemi yaptığımız için halüsinasyonu önlemek adına 0.0 yapıldı.
        })
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            return f"[HATA: {response.status_code}]"
    except Exception as e:
        return f"[HATA: Bağlantı sorunu - {str(e)}]"

def calculate_similarity(text1, text2):
    """İki metin arasındaki benzerliği 0.0 ile 1.0 arasında döndürür."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def main():
    print("Veri seti yükleniyor...")
    ds = load_from_disk(r'C:\work environment\Python\nlp_exercises\saved_dataset')
    df = ds['train'].to_pandas()

    print("Regex işlemi uygulanıyor...")
    df['hukum_text_regex'] = df['text'].apply(extract_verdict_pure)
    successful_extractions = df[df['hukum_text_regex'].notna()]
    
    # Sabit 100 örnek seçimi (her çalıştırmada aynı örnekler seçilsin diye random_state sabit)
    sample_size = min(100, len(successful_extractions))
    df_sample = successful_extractions.sample(n=sample_size, random_state=42).copy()
    
    # Çıktı klasörünü oluştur
    output_dir = "consistency_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tüm LLM sonuçlarını tutacağımız bir sözlük: {id: [run1, run2, run3, run4, run5]}
    llm_runs_history = {row['id']: [] for _, row in df_sample.iterrows()}
    
    num_runs = 5
    for run_idx in range(1, num_runs + 1):
        print(f"\n--- ÇALIŞTIRMA {run_idx}/{num_runs} BAŞLADI ---")
        run_results = []
        
        # tqdm kullanmadan basit sayac
        total_items = len(df_sample)
        for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
            print(f"  [Tur {run_idx}] Örnek {i}/{total_items} işleniyor...", end='\r')
            llm_output = extract_verdict_with_llm(row['text'])
            run_results.append(llm_output)
            llm_runs_history[row['id']].append(llm_output)
            
        print(f"\n  [Tur {run_idx}] Tüm örnekler işlendi. Kaydediliyor...")
        # Bu iterasyonun sonuçlarını kaydet
        df_run = df_sample[['id', 'text', 'hukum_text_regex']].copy()
        df_run['text_tail_preview'] = df_run['text'].apply(lambda x: "..." + x[-500:].replace('\n', ' '))
        df_run['hukum_text_llm'] = run_results
        
        excel_path = os.path.join(output_dir, f"llm_run_{run_idx}.xlsx")
        df_run[['id', 'text_tail_preview', 'hukum_text_regex', 'hukum_text_llm']].to_excel(excel_path, index=False)
        print(f"Çalıştırma {run_idx} tamamlandı ve '{excel_path}' konumuna kaydedildi.")
        
    print("\n--- BENZERLİK HESAPLAMASI (CONSISTENCY) ---")
    consistency_scores = []
    
    for _, row in df_sample.iterrows():
        record_id = row['id']
        outputs = llm_runs_history[record_id]
        
        # 5 çıktı arasından ikili kombinasyonları al (toplam 10 çift)
        pairs = list(combinations(outputs, 2))
        similarities = [calculate_similarity(t1, t2) for t1, t2 in pairs]
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        consistency_scores.append(avg_similarity)
        
    # Final Rapor Dataframe'i
    df_sample['avg_consistency_score'] = consistency_scores
    for i in range(num_runs):
        df_sample[f'run_{i+1}_output'] = [llm_runs_history[row['id']][i] for _, row in df_sample.iterrows()]
        
    final_report_path = os.path.join(output_dir, "final_consistency_report.xlsx")
    
    # Kolonları düzenle
    cols_to_save = ['id', 'hukum_text_regex', 'avg_consistency_score'] + [f'run_{i+1}_output' for i in range(num_runs)]
    df_sample[cols_to_save].to_excel(final_report_path, index=False)
    
    print(f"\nTüm işlemler bitti! Rapor '{final_report_path}' dosyasına kaydedildi.")
    print(f"Genel Ortalama Benzerlik Skoru (Tüm 100 örnek için): {df_sample['avg_consistency_score'].mean():.4f}")

if __name__ == "__main__":
    main()
