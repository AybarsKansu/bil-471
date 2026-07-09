import os
import json
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def prepare_finetune_data():
    print("Kategorizasyon sonuçları (etiketler) yükleniyor...")
    with open('categorization_results.json', 'r', encoding='utf-8') as f:
        cat_results = json.load(f)
    
    data_mapping = cat_results.get('data_mapping', {})
    print(f"Toplam etiketlenmiş veri sayısı: {len(data_mapping)}")

    print("\nEğitim (Train - Regex) verisi yükleniyor...")
    # 'dataset_train' contains the items with regex extractions
    train_ds = load_from_disk('dataset_train')
    df_train = train_ds.to_pandas()
    
    # Not all data might have labels if only a subset was categorized.
    # Let's filter df_train to only include IDs present in data_mapping
    df_train = df_train[df_train['id'].astype(str).isin(data_mapping.keys())].copy()
    
    # Etiketleri ekle
    df_train['label'] = df_train['id'].astype(str).map(data_mapping)
    
    # Regex ile çıkarılmış hüküm metnini text olarak al (yoksa normal text'i)
    if 'hukum_text_regex' in df_train.columns:
        df_train['text'] = df_train['hukum_text_regex']
    else:
        print("Uyarı: 'hukum_text_regex' kolonu bulunamadı, varsayılan 'text' kullanılacak.")
        
    df_train = df_train[['id', 'text', 'label']].dropna(subset=['text', 'label'])
    
    # Train ve test olarak ayır (Train %90, Test %10)
    train_df, test_regex_df = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['label'])
    print(f"Regex Eğitim Verisi Boyutu: {len(train_df)}")
    print(f"Regex Test Verisi Boyutu: {len(test_regex_df)}")

    print("\nAltın Veri Seti (Gold Test - LLM) yükleniyor...")
    gold_file = os.path.join('async_extraction_outputs', 'checkpoint_results.json')
    if os.path.exists(gold_file):
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_results = json.load(f)
        
        gold_records = []
        for doc_id, text in gold_results.items():
            if str(doc_id) in data_mapping:
                gold_records.append({
                    'id': str(doc_id),
                    'text': text,
                    'label': data_mapping[str(doc_id)]
                })
        
        gold_df = pd.DataFrame(gold_records)
        print(f"Altın Test Verisi Boyutu: {len(gold_df)}")
        
        # Test setini Regex test verisi ile birleştir
        final_test_df = pd.concat([test_regex_df, gold_df], ignore_index=True)
    else:
        print(f"{gold_file} bulunamadı, sadece regex test verisi kullanılacak.")
        final_test_df = test_regex_df

    # DatasetDict formatına dönüştür ve kaydet
    final_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(final_test_df.reset_index(drop=True))
    })

    output_dir = 'dataset_finetune_ready'
    final_dataset.save_to_disk(output_dir)
    print(f"\nVeri seti başarıyla hazırlandı ve '{output_dir}' klasörüne kaydedildi.")
    print(final_dataset)

if __name__ == "__main__":
    prepare_finetune_data()
