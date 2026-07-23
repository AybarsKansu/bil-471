import json
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parent
LABEL_FILE = REPO_ROOT / "categorization_results.json"
TRAIN_DATASET_PATH = REPO_ROOT / "dataset_train"
GOLD_DATASET_PATH = REPO_ROOT / "dataset_gold_test"
GOLD_LLM_FILE = REPO_ROOT / "async_extraction_outputs" / "checkpoint_results.json"
OUTPUT_DIR = REPO_ROOT / "dataset_finetune_ready"
QUALITY_REPORT = REPO_ROOT / "dataset_finetune_quality_report.csv"

VERDICT_HEADING_RE = re.compile(
    r"(?i)(?:^|\n|[.!?]\s+)"
    r"(?P<heading>"
    r"(?:SONUÇ|S\s*O\s*N\s*U\s*Ç|"
    r"HÜKÜM|H\s*Ü\s*K\s*Ü\s*M|"
    r"KARAR|K\s*A\s*R\s*A\s*R|"
    r"NETİCE|GEREĞİ DÜŞÜNÜLDÜ)\s*[:;]?"
    r")"
)
TRAILING_DECISION_CUE_RE = re.compile(
    r"(?i)("
    r"bozulmalı|bozulmasına|onanmalı|onanmasına|"
    r"bozulmamalı|onanmalıdır|bozulmalıdır|"
    r"reddedilmeli|reddine|kabul edilmeli|kabul edilmelidir|kabulüne|"
    r"düzeltilerek|ortadan kaldırılmasına|gönderilmesine|tevdiine|"
    r"bozma nedenidir|hatalı olmuştur|yerinde değildir|isabetsizdir|"
    r"bozmayı gerektirmiştir|bozulması gerekmiştir|"
    r"usul ve yasaya aykırı|karar vermek gerekmiştir|karar verildi|"
    r"yerinde görülmemiştir|doğru görülmemiştir|reddi gerekmiştir|"
    r"incelenmesine gerek görülmemiştir|"
    r"oybirliğiyle|oyçokluğuyla"
    r")"
)


def normalize_text(value):
    text = "" if value is None or pd.isna(value) else str(value)
    text = text.replace("\xa0", " ").replace("**", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_verdict_from_text(full_text, verdict_text):
    full_text = normalize_text(full_text)
    verdict_text = normalize_text(verdict_text)
    if not full_text or not verdict_text:
        cleaned_text, cues_removed = strip_trailing_verdict_cues(full_text)
        return cleaned_text, False, cues_removed

    start = full_text.rfind(verdict_text)
    if start == -1:
        cleaned_text, cues_removed = strip_trailing_verdict_cues(full_text)
        return cleaned_text, False, cues_removed

    masked_text = full_text[:start].strip()
    cleaned_text, cues_removed = strip_trailing_verdict_cues(masked_text)
    return cleaned_text, True, cues_removed


def strip_trailing_verdict_cues(text):
    text = normalize_text(text)
    if not text:
        return text, False

    changed = False
    for match in reversed(list(VERDICT_HEADING_RE.finditer(text))):
        heading_start = match.start("heading")
        if heading_start >= len(text) * 0.4 or len(text) - heading_start <= 8000:
            text = text[:heading_start].strip()
            changed = True
            break

    paragraphs = re.split(r"\n\s*\n", text)
    while paragraphs and TRAILING_DECISION_CUE_RE.search(paragraphs[-1]):
        paragraphs.pop()
        changed = True

    text = "\n\n".join(paragraphs).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    while sentences and TRAILING_DECISION_CUE_RE.search(sentences[-1]):
        sentences.pop()
        changed = True

    text = " ".join(sentences).strip()
    return text, changed


def build_labeled_frame(df, data_mapping, text_source_name):
    frame = df[df["id"].astype(str).isin(data_mapping.keys())].copy()
    frame["label"] = frame["id"].astype(str).map(data_mapping)

    if "hukum_text_regex" not in frame.columns:
        raise KeyError("Expected column 'hukum_text_regex' in prepared datasets.")

    masked = frame.apply(
        lambda row: remove_verdict_from_text(row["text"], row["hukum_text_regex"]),
        axis=1,
        result_type="expand",
    )
    frame["model_input_text"] = masked[0]
    frame["verdict_removed"] = masked[1]
    frame["tail_cues_removed"] = masked[2]
    frame["verdict_text"] = frame["hukum_text_regex"].map(normalize_text)
    frame["source"] = text_source_name
    frame["input_char_len"] = frame["model_input_text"].str.len()
    frame["verdict_char_len"] = frame["verdict_text"].str.len()
    frame["possible_leakage"] = frame.apply(
        lambda row: bool(row["verdict_text"]) and row["verdict_text"] in row["model_input_text"],
        axis=1,
    )

    frame["text"] = frame["model_input_text"]
    return frame[
        [
            "id",
            "text",
            "label",
            "source",
            "verdict_text",
            "verdict_removed",
            "tail_cues_removed",
            "possible_leakage",
            "input_char_len",
            "verdict_char_len",
        ]
    ].dropna(subset=["text", "label"])


def prepare_finetune_data():
    print("Kategorizasyon sonuçları (etiketler) yükleniyor...")
    with LABEL_FILE.open("r", encoding="utf-8") as f:
        cat_results = json.load(f)

    data_mapping = cat_results.get("data_mapping", {})
    print(f"Toplam etiketlenmiş veri sayısı: {len(data_mapping)}")

    print("\nEğitim verisi yükleniyor ve hüküm kısmı maskeleniyor...")
    train_ds = load_from_disk(str(TRAIN_DATASET_PATH))
    train_df_raw = train_ds.to_pandas()
    labeled_train_df = build_labeled_frame(train_df_raw, data_mapping, "train_regex_masked")

    labeled_train_df = labeled_train_df[labeled_train_df["text"].str.len() > 0].copy()
    train_df, test_regex_df = train_test_split(
        labeled_train_df,
        test_size=0.1,
        random_state=42,
        stratify=labeled_train_df["label"],
    )
    print(f"Masked eğitim verisi boyutu: {len(train_df)}")
    print(f"Masked regex test verisi boyutu: {len(test_regex_df)}")

    final_test_df = test_regex_df.copy()

    print("\nGold test verisi kontrol ediliyor...")
    if GOLD_DATASET_PATH.exists() and GOLD_LLM_FILE.exists():
        gold_ds = load_from_disk(str(GOLD_DATASET_PATH))
        gold_raw_df = gold_ds.to_pandas()
        gold_df = build_labeled_frame(gold_raw_df, data_mapping, "gold_regex_masked")

        with GOLD_LLM_FILE.open("r", encoding="utf-8") as f:
            gold_results = json.load(f)

        gold_df["llm_verdict_text"] = gold_df["id"].astype(str).map(gold_results)
        print(f"Gold test verisi boyutu: {len(gold_df)}")
        final_test_df = pd.concat([final_test_df, gold_df], ignore_index=True)
    else:
        print("Gold LLM çıktısı bulunamadı, sadece regex test verisi kullanılacak.")

    report_df = pd.concat([train_df, final_test_df], ignore_index=True)
    quality_sample = report_df[
        [
            "id",
            "source",
            "label",
            "verdict_removed",
            "tail_cues_removed",
            "possible_leakage",
            "input_char_len",
            "verdict_char_len",
            "text",
            "verdict_text",
        ]
    ].head(500).rename(
        columns={
            "text": "model_input_text",
            "verdict_text": "removed_verdict_text",
        }
    )
    quality_sample.to_csv(QUALITY_REPORT, index=False, encoding="utf-8-sig")

    print("\nKalite özeti:")
    print(report_df.groupby("source").size())
    print("Hüküm kaldırma oranı:", round(report_df["verdict_removed"].mean(), 4))
    print("Olası leakage oranı:", round(report_df["possible_leakage"].mean(), 4))
    print("Label dağılımı:")
    print(report_df["label"].value_counts())
    print(f"İlk 500 örnek kalite raporu: {QUALITY_REPORT}")

    train_output = train_df[["id", "text", "label", "source"]].reset_index(drop=True)
    test_output = final_test_df[["id", "text", "label", "source"]].reset_index(drop=True)

    final_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_output),
        "test": Dataset.from_pandas(test_output),
    })

    final_dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"\nVeri seti başarıyla hazırlandı ve '{OUTPUT_DIR}' klasörüne kaydedildi.")
    print(final_dataset)


if __name__ == "__main__":
    prepare_finetune_data()
