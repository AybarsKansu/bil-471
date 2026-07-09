import json
import os

def merge_categories(file_path="categorization_results.json"):
    if not os.path.exists(file_path):
        print(f"Hata: '{file_path}' bulunamadı.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Adım 1: İçinde "Temyiz" geçen tüm isimleri "Temyiz Davası" olarak güncelle
    for cluster_id, name in data.get("cluster_names", {}).items():
        if "Temyiz" in name:
            data["cluster_names"][cluster_id] = "Temyiz Davası"

    # Adım 2: Aynı isme sahip olan cluster_id'leri grupla
    name_to_keys = {}
    for cluster_id, name in data.get("cluster_names", {}).items():
        name_to_keys.setdefault(name, []).append(cluster_id)

    # Değişiklik gerekip gerekmediğini kontrol et (hiçbir grupta 1'den fazla key yoksa gerek yok)
    needs_merge = any(len(keys) > 1 for keys in name_to_keys.values())

    if not needs_merge:
        print("\nBirleştirilecek yeni bir kategori bulunamadı. (Kümeler ve ID'ler zaten tekilleştirilmiş)")
        return

    # Adım 3: Temiz bir sıralamayla (0, 1, 2, 3...) yeni kümeleri ve ID haritasını oluştur
    new_cluster_names = {}
    old_to_new_id = {}
    
    for i, (name, keys) in enumerate(name_to_keys.items()):
        new_cluster_names[str(i)] = name
        for old_key in keys:
            old_to_new_id[str(old_key)] = i

    # Adım 4: Verilerdeki eski ID'leri yeni tekilleştirilmiş ID'lerle değiştir
    for data_id, old_cluster_id in data.get("data_mapping", {}).items():
        new_id = old_to_new_id[str(old_cluster_id)]
        data["data_mapping"][data_id] = new_id

    # Adım 5: Yeni cluster_names'i set et
    data["cluster_names"] = new_cluster_names

    # Adım 6: Dosyayı kaydet
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print("\nBirleştirme işlemi başarıyla tamamlandı! Aynı isme sahip kümeler tek bir ID altında toplandı ve dosya kaydedildi.")

if __name__ == "__main__":
    merge_categories()
