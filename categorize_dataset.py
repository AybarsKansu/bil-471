import os
import json
import argparse
import re
import requests
import time
import pandas as pd
from pathlib import Path
from datasets import load_from_disk, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
DEFAULT_NUM_CLUSTERS = int(os.environ.get("NUM_CLUSTERS", "10"))
REPO_ROOT = Path(__file__).resolve().parent
CATEGORY_OUTPUT_DIR = REPO_ROOT / "category_outputs"
CLUSTER_RAW_RESPONSES = {}

def infer_category_name_from_samples(samples, cluster_id):
    joined = " ".join(str(sample) for sample in samples).lower()

    if "yargi yeri" in joined or "yargı yeri" in joined or "yargä± yeri" in joined:
        return "Yargı Yeri Belirleme"

    if "yerel mahkemeye" in joined and ("gönder" in joined or "gã–nder" in joined or "gã¶nder" in joined):
        return "Eksiklik Giderme"

    if "gönderilmesine" in joined or "gã–nder" in joined or "gã¶nder" in joined or "tevdi" in joined:
        return "Dosya Gönderimi"

    has_bozma = "bozulmasina" in joined or "bozulmasına" in joined
    has_onama = "onanmasina" in joined or "onanmasına" in joined

    if has_bozma and has_onama:
        return "Kısmi Bozma Onama"
    if has_onama:
        return "Onama"
    if has_bozma and ("incelenmesine yer olmad" in joined or "incelenmesine yer olmadä" in joined):
        return "Bozma İnceleme Yok"
    if has_bozma:
        return "Bozma"
    if "reddine" in joined or "reddä°ne" in joined:
        return "Ret"

    return f"Kategori_{cluster_id}"

def clean_category_name(raw_response, cluster_id):
    category_name = str(raw_response or "").strip()
    category_name = re.sub(r"<think>.*?</think>", "", category_name, flags=re.DOTALL | re.IGNORECASE).strip()
    category_name = category_name.replace('"', '').replace("'", "")
    category_name = category_name.replace("Kategori:", "").replace("Kategori adı:", "").strip()

    non_empty_lines = [line.strip(" -\t") for line in category_name.splitlines() if line.strip(" -\t")]
    if non_empty_lines:
        category_name = non_empty_lines[0]

    if not category_name:
        return f"Kategori_{cluster_id}"

    return category_name[:80]

def get_category_name_from_llm(samples, cluster_id):
    """
    Belirli bir kümeye (cluster) ait 10 rastgele örneği LLM'e yollayıp
    bu kümenin hangi hukuki kategoriye girdiğini sorar.
    """
    text_samples = "\n\n---\n\n".join([f"Örnek {i+1}: {text}" for i, text in enumerate(samples)])
    
    prompt = f"""/no_think
Aşağıda aynı kümeye ait mahkeme kararlarının HÜKÜM/SONUÇ kısımlarından örnekler var.
Görevin bu kümeye kısa ve ayırt edici bir hukuki kategori adı vermek.

Kurallar:
- Sadece 1-4 kelimelik kategori adı yaz.
- Açıklama, gerekçe, madde numarası, tırnak veya liste kullanma.
- Mümkünse usul sonucunu değil hukuki alanı/dava türünü adlandır.
- "Temyiz", "Bozma", "Onama", "Ret", "Mahkeme", "Karar" gibi genel usul kelimelerini tek başına kategori adı yapma.
- Eğer örnekler gerçekten sadece usul/kanun yolu sonucu içeriyorsa daha ayırt edici bir ad kullan: "Kanun Yolu", "Görev Uyuşmazlığı", "Dosya Gönderimi" gibi.
- Ceza, iş, aile, tazminat, icra, vergi, idare, miras, kira, tüketici, kamulaştırma gibi alanlardan biri belirginse onu tercih et.

İyi cevap örnekleri:
Ceza Davası
İşçilik Alacağı
Boşanma Davası
Kamulaştırma
İdari Dava
Dosya Gönderimi

Kötü cevap örnekleri:
Temyiz Davası
Mahkeme Kararı
Karar Sonucu
Hukuki Kategori

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
                raw_response = response.json().get('response', '').strip()
                CLUSTER_RAW_RESPONSES[str(cluster_id)] = raw_response
                category_name = clean_category_name(raw_response, cluster_id)
                if category_name == f"Kategori_{cluster_id}":
                    category_name = infer_category_name_from_samples(samples, cluster_id)
                    print(f"Cluster {cluster_id} icin model bos cevap verdi; otomatik ad kullaniliyor: {category_name}")
                return category_name
            else:
                time.sleep(2)
        except Exception as e:
            if attempt == 2:
                print(f"Cluster {cluster_id} isimlendirmesinde hata: {e}")
            time.sleep(2)
            
    return f"Kategori_{cluster_id}"

def write_category_reports(df, cluster_names):
    CATEGORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assignment_columns = [
        "global_index",
        "dataset_source",
        "split_index",
        "id",
        "cluster_id",
        "category",
    ]
    available_columns = [col for col in assignment_columns if col in df.columns]
    assignments = df[available_columns].copy()

    assignments_csv = CATEGORY_OUTPUT_DIR / "category_assignments.csv"
    assignments_txt = CATEGORY_OUTPUT_DIR / "category_assignments.txt"
    summary_txt = CATEGORY_OUTPUT_DIR / "category_summary.txt"
    cluster_names_json = CATEGORY_OUTPUT_DIR / "cluster_names.json"
    raw_responses_json = CATEGORY_OUTPUT_DIR / "cluster_llm_raw_responses.json"

    assignments.to_csv(assignments_csv, index=False, encoding="utf-8-sig")

    with assignments_txt.open("w", encoding="utf-8") as f:
        f.write("global_index\tdataset_source\tsplit_index\tid\tcluster_id\tcategory\n")
        for row in assignments.itertuples(index=False):
            f.write("\t".join(str(value) for value in row) + "\n")

    with cluster_names_json.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cluster_names.items()}, f, ensure_ascii=False, indent=2)

    with raw_responses_json.open("w", encoding="utf-8") as f:
        json.dump(CLUSTER_RAW_RESPONSES, f, ensure_ascii=False, indent=2)

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("CATEGORY DISTRIBUTION\n")
        f.write("=====================\n")
        f.write(df["category"].value_counts().to_string())
        f.write("\n\nCLUSTER DISTRIBUTION\n")
        f.write("====================\n")
        for cluster_id in sorted(cluster_names):
            cluster_df = df[df["cluster_id"] == cluster_id]
            f.write(
                f"cluster_id={cluster_id}\tcategory={cluster_names[cluster_id]}"
                f"\tcount={len(cluster_df)}\n"
            )
            for _, row in cluster_df.head(5).iterrows():
                preview = str(row.get("hukum_text_regex", "")).replace("\n", " ")[:240]
                f.write(
                    f"  global_index={row['global_index']}"
                    f"\tdataset_source={row['dataset_source']}"
                    f"\tsplit_index={row['split_index']}"
                    f"\tid={row.get('id', '')}"
                    f"\tpreview={preview}\n"
                )
            f.write("\n")

    print("\nKategori raporlari kaydedildi:")
    print(f"- {summary_txt}")
    print(f"- {assignments_txt}")
    print(f"- {assignments_csv}")
    print(f"- {cluster_names_json}")
    print(f"- {raw_responses_json}")

def main(num_clusters):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Veri setleri (Test ve Train) yükleniyor...")
    # Ayrılmış veri setlerini yüklüyoruz
    ds_test = load_from_disk(str(REPO_ROOT / "dataset_gold_test"))
    ds_train = load_from_disk(str(REPO_ROOT / "dataset_train"))
    
    # Veri setlerini birleştirirken kaynak ekleme
    df_test = ds_test.to_pandas()
    df_test['dataset_source'] = 'test'
    df_test['split_index'] = range(len(df_test))
    
    df_train = ds_train.to_pandas()
    df_train['dataset_source'] = 'train'
    df_train['split_index'] = range(len(df_train))
    
    df = pd.concat([df_test, df_train], ignore_index=True)
    df['global_index'] = df.index
    
    print(f"Toplam {len(df)} kayıt üzerinde otomatik kategorizasyon başlatılıyor...")
    
    # 1. Hızlı Vektörizasyon (TF-IDF)
    # n_features=10000, max_df=0.9, min_df=5 gibi ayarlar çok hızlı ve doğru kümeleme sağlar
    print("Metinler vektörlere dönüştürülüyor (TF-IDF)...")
    turkish_stop_words = ["ve", "veya", "ile", "için", "gibi", "kadar", "olan", "olarak", "dava", "davası", "davaya", "karar", "hüküm", "tarih", "madde", "göre", "tarafından", "hakkında", "sayılı", "mahkemesi"]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=turkish_stop_words, ngram_range=(1, 2), max_df=0.8, min_df=10)
    # Hüküm metni üzerinden konuları bulmak daha sağlıklıdır, çünkü gereksiz gürültü yoktur.
    X = vectorizer.fit_transform(df['hukum_text_regex'])
    
    # 2. Doğrudan K-Means
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
    write_category_reports(df, cluster_names)
    
    # 5. Sonuçları Geri Kaydetme
    # Datayı bölündüğü gibi tekrar ayırıp kaydedelim
    print("\nEtiketlenmiş veri setleri diske kaydediliyor...")
    
    df_test_final = df[df['dataset_source'] == 'test'].drop(columns=['dataset_source'])
    df_train_final = df[df['dataset_source'] == 'train'].drop(columns=['dataset_source'])
    
    test_dataset_labeled = Dataset.from_pandas(df_test_final.reset_index(drop=True))
    train_dataset_labeled = Dataset.from_pandas(df_train_final.reset_index(drop=True))
    
    test_dataset_labeled.save_to_disk(str(REPO_ROOT / "dataset_gold_test_categorized"))
    train_dataset_labeled.save_to_disk(str(REPO_ROOT / "dataset_train_categorized"))
    
    print("Tüm işlemler başarıyla tamamlandı. Kategorize edilmiş veri setleri klasörlere kaydedildi:")
    print("- dataset_gold_test_categorized")
    print("- dataset_train_categorized")

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster and label the prepared legal datasets.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--num-clusters", type=int, default=DEFAULT_NUM_CLUSTERS)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model
    main(args.num_clusters)
