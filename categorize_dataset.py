import os
import requests
import time
import pandas as pd
from datasets import load_from_disk, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"

def get_category_name_from_llm(samples, cluster_id):
    """
    Belirli bir kümeye (cluster) ait 10 rastgele örneği LLM'e yollayıp
    bu kümenin hangi hukuki kategoriye girdiğini sorar.
    """
    text_samples = "\n\n---\n\n".join([f"Örnek {i+1}: {text}" for i, text in enumerate(samples)])
    
    prompt = f"""Aşağıda aynı hukuki konuya (kategoriye) ait 10 farklı mahkeme kararının sonuç kısımları verilmiştir.
Lütfen bu kararların ortak noktasını bularak bu gruba 1-3 kelimelik kısa bir KATEGORİ İSMİ ver. 
Örneğin: 'Ceza Davası', 'Boşanma Davası', 'İş Hukuku', 'Tazminat Davası' gibi. 
Sadece kategori ismini yaz, başka hiçbir açıklama ekleme.

{text_samples}"""

    for attempt in range(3):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 50}
            })
            if response.status_code == 200:
                category_name = response.json().get('response', '').strip()
                category_name = category_name.replace('"', '').replace("'", "")
                category_name = category_name.replace("Kategori:", "").strip()
                return category_name
            else:
                time.sleep(2)
        except Exception as e:
            if attempt == 2:
                print(f"Cluster {cluster_id} isimlendirmesinde hata: {e}")
            time.sleep(2)
            
    return f"Kategori_{cluster_id}"

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Veri setleri (Test ve Train) yükleniyor...")
    # Ayrılmış veri setlerini yüklüyoruz
    ds_test = load_from_disk(r'C:\work environment\Python\bil 471\dataset_gold_test')
    ds_train = load_from_disk(r'C:\work environment\Python\bil 471\dataset_train')
    
    # Veri setlerini birleştirirken kaynak ekleme
    df_test = ds_test.to_pandas()
    df_test['dataset_source'] = 'test'
    
    df_train = ds_train.to_pandas()
    df_train['dataset_source'] = 'train'
    
    df = pd.concat([df_test, df_train], ignore_index=True)
    
    print(f"Toplam {len(df)} kayıt üzerinde otomatik kategorizasyon başlatılıyor...")
    
    # 1. Hızlı Vektörizasyon (TF-IDF)
    # n_features=10000, max_df=0.9, min_df=5 gibi ayarlar çok hızlı ve doğru kümeleme sağlar
    print("Metinler vektörlere dönüştürülüyor (TF-IDF)...")
    turkish_stop_words = ["ve", "veya", "ile", "için", "gibi", "kadar", "olan", "olarak", "dava", "davası", "davaya", "karar", "hüküm", "tarih", "madde", "göre", "tarafından", "hakkında", "sayılı", "mahkemesi"]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=turkish_stop_words, ngram_range=(1, 2), max_df=0.8, min_df=10)
    # Hüküm metni üzerinden konuları bulmak daha sağlıklıdır, çünkü gereksiz gürültü yoktur.
    X = vectorizer.fit_transform(df['hukum_text_regex'])
    
    # 2. Doğrudan K-Means (10 Kategori)
    num_clusters = 10
    
    print(f"\nSeçilen K={num_clusters} değeri ile nihai kümeleme yapılıyor...")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=2048)
    df['cluster_id'] = kmeans.fit_predict(X)
    
    # 3. Kümeleri LLM ile İsimlendirme
    print("\nOluşturulan kümeler LLM kullanılarak isimlendiriliyor...")
    cluster_names = {}
    
    for cluster_id in range(num_clusters):
        # Bu kümeye ait rastgele örnek seç
        cluster_df = df[df['cluster_id'] == cluster_id]['hukum_text_regex']
        sample_size = min(10, len(cluster_df))
        cluster_samples = cluster_df.sample(n=sample_size, random_state=42).tolist()
        
        # LLM'e sor
        print(f"Küme {cluster_id} için LLM'den isim bekleniyor...")
        category_name = get_category_name_from_llm(cluster_samples, cluster_id)
        cluster_names[cluster_id] = category_name
        safe_name = category_name.encode('cp1254', 'replace').decode('cp1254')
        print(f"  -> Küme {cluster_id} İsimlendirildi: {safe_name}")
        
    # 4. İsimleri Dataframe'e uygula
    df['category'] = df['cluster_id'].map(cluster_names)
    
    print("\nKategorizasyon tamamlandı! Örnek dağılım:")
    print(df['category'].value_counts())
    
    # 5. Sonuçları Geri Kaydetme
    # Datayı bölündüğü gibi tekrar ayırıp kaydedelim
    print("\nEtiketlenmiş veri setleri diske kaydediliyor...")
    
    df_test_final = df[df['dataset_source'] == 'test'].drop(columns=['dataset_source'])
    df_train_final = df[df['dataset_source'] == 'train'].drop(columns=['dataset_source'])
    
    test_dataset_labeled = Dataset.from_pandas(df_test_final.reset_index(drop=True))
    train_dataset_labeled = Dataset.from_pandas(df_train_final.reset_index(drop=True))
    
    test_dataset_labeled.save_to_disk(r'C:\work environment\Python\bil 471\dataset_gold_test_categorized')
    train_dataset_labeled.save_to_disk(r'C:\work environment\Python\bil 471\dataset_train_categorized')
    
    print("Tüm işlemler başarıyla tamamlandı. Kategorize edilmiş veri setleri klasörlere kaydedildi:")
    print("- dataset_gold_test_categorized")
    print("- dataset_train_categorized")

if __name__ == "__main__":
    main()
