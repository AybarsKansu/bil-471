import os
import re
import json
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio
import warnings
warnings.filterwarnings('ignore')

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

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"
CONCURRENCY_LIMIT = 64
REPO_ROOT = Path(__file__).resolve().parent

async def fetch_llm_result(session, sem, record_id, text, retries=3):
    prompt = f"Aşağıdaki hukuki metnin sadece hüküm (sonuç/karar) kısmını aynen yaz. Ekstra hiçbir yorum veya açıklama ekleme:\n\n{text[-2000:]}"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    async with sem:
        for attempt in range(retries):
            try:
                # Timeout süresi eklendi (tek bir istek sonsuza kadar asılı kalmasın diye)
                async with session.post(OLLAMA_URL, json=payload, timeout=120) as response:
                    if response.status == 200:
                        data = await response.json()
                        return record_id, data.get('response', '').strip()
                    else:
                        await asyncio.sleep(2)
            except Exception as e:
                if attempt == retries - 1:
                    return record_id, f"[HATA: {str(e)}]"
                await asyncio.sleep(2)
        return record_id, "[HATA: Max retries ulaşıldı]"

async def main_async():
    # 1. Kategorize Edilmiş Test Veri Setini Yükleme
    print("Test veri seti (15.000 Gold) yükleniyor...")
    ds = load_from_disk(str(REPO_ROOT / "dataset_gold_test_categorized"))

    # ARKADAŞINIZLA BÖLÜŞMEK İÇİN AYARLAR (Yarı yarıya bölüşmek için)
    # Sizin PC için -> START_INDEX = 0, END_INDEX = 7500
    # Arkadaşınızın PC için -> START_INDEX = 7500, END_INDEX = 15000
    START_INDEX = 0
    END_INDEX = 15000  # Hepsini tek PC'de yapmak için 15000 kalsın
    df_filtered = ds.to_pandas().iloc[START_INDEX:END_INDEX].copy()

    total_samples = len(df_filtered)
    print(f"Toplam {total_samples} adet test örneği Qwen2.5-14B ile işlenecek.")

    output_dir = REPO_ROOT / "async_extraction_outputs"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint_results.json"

    # 2. Kaldığı yerden devam etme (Checkpointing)
    completed_results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            completed_results = json.load(f)
        print(f"Önceki çalıştırmadan {len(completed_results)} adet sonuç yüklendi.")

    tasks_to_run = []
    for idx, row in df_filtered.iterrows():
        record_id = str(row['id'])
        if record_id not in completed_results:
            tasks_to_run.append((record_id, row['text']))

    print(f"Sıradaki işlenecek kayıt sayısı: {len(tasks_to_run)}")
    if len(tasks_to_run) == 0:
        print("İşlenecek kayıt kalmadı. Tüm işlemler tamam!")
        return

    # 3. Asenkron İstek Havuzu
    # Terminalde OLLAMA_NUM_PARALLEL ayarlı olmalı (Set OLLAMA_NUM_PARALLEL=8 gibi)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_llm_result(session, sem, record_id, text) for record_id, text in tasks_to_run]

        # tqdm_asyncio ekranda muazzam bir ilerleme çubuğu gösterecek
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Ollama Paralel İşleniyor"):
            record_id, llm_output = await f
            completed_results[record_id] = llm_output

            # Her 100 işlemde bir diske yaz, hata olursa gitmesin
            if len(completed_results) % 100 == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as cf:
                    json.dump(completed_results, cf, ensure_ascii=False, indent=2)

    # 4. İşlem Sonu ve Final Excel Çıktısı
    with open(checkpoint_file, 'w', encoding='utf-8') as cf:
        json.dump(completed_results, cf, ensure_ascii=False, indent=2)

    print("\n[TAMAMLANDI] Tüm API istekleri bitti.")

    df_final = df_filtered.copy()
    df_final['hukum_text_llm'] = df_final['id'].astype(str).map(completed_results)

    final_excel = output_dir / "final_parallel_extraction.xlsx"
    df_final.to_excel(final_excel, index=False)
    print(f"Sonuçlar başarıyla '{final_excel}' konumuna kaydedildi!")

if __name__ == "__main__":
    # Windows ortamında aiohttp / asyncio hatalarını gidermek için:
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_async())
