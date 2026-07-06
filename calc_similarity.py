import os
import pandas as pd
from difflib import SequenceMatcher

def calculate_similarity(text1, text2):
    if pd.isna(text1): text1 = ""
    if pd.isna(text2): text2 = ""
    text1, text2 = str(text1).strip(), str(text2).strip()
    if not text1 or not text2: return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def main():
    output_dir = r"c:\work environment\Python\bil 471\consistency_outputs"
    target_file = os.path.join(output_dir, "llm_run_1.xlsx")
    
    if not os.path.exists(target_file):
        print(f"Hata: {target_file} bulunamadı.")
        return
        
    print(f"Dosya okunuyor: {target_file}")
    df = pd.read_excel(target_file)
    
    print("Regex çıktıları ile LLM çıktıları karşılaştırılıyor...")
    
    similarities = []
    for idx, row in df.iterrows():
        regex_text = row['hukum_text_regex']
        llm_text = row['hukum_text_llm']
        score = calculate_similarity(regex_text, llm_text)
        similarities.append(score)
        
    df['regex_vs_llm_similarity'] = similarities
    
    mean_score = df['regex_vs_llm_similarity'].mean()
    print(f"\nİşlem tamamlandı! {len(df)} adet örnek kıyaslandı.")
    print(f"--> Regex ile LLM Arası Ortalama Benzerlik Skoru: {mean_score:.4f}")
    
    out_path = os.path.join(output_dir, "regex_vs_llm_report.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Detaylı sonuçlar '{out_path}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
