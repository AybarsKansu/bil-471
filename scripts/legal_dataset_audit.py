import argparse
import csv
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_from_disk


REPO_ROOT = Path(__file__).resolve().parents[1]

TRANSLIT_TABLE = str.maketrans(
    {
        "c": "c",
        "g": "g",
        "i": "i",
        "o": "o",
        "s": "s",
        "u": "u",
        "C": "c",
        "G": "g",
        "I": "i",
        "O": "o",
        "S": "s",
        "U": "u",
        "\u00e7": "c",
        "\u011f": "g",
        "\u0131": "i",
        "\u00f6": "o",
        "\u015f": "s",
        "\u00fc": "u",
        "\u00c7": "c",
        "\u011e": "g",
        "\u0130": "i",
        "\u00d6": "o",
        "\u015e": "s",
        "\u00dc": "u",
    }
)

SECTION_HEADER_PATTERNS = [
    re.compile(r"(?m)^\s*(hukum|sonuc|karar sonucu|hukum sonucu)\s*[:\-]?\s*$"),
    re.compile(r"(?m)^\s*(hukum:)"),
    re.compile(r"(?m)^\s*(sonuc:)"),
]

CATEGORY_PATTERNS = {
    "beraat": [
        re.compile(r"\bberaat\b"),
    ],
    "hapis": [
        re.compile(r"\bhapis\b"),
        re.compile(r"\bhurriyeti baglayici\b"),
    ],
    "adli_para_cezasi": [
        re.compile(r"\badli para ceza"),
        re.compile(r"\bpara ceza"),
    ],
    "hagb": [
        re.compile(r"\bhukmun aciklanmasinin geri birak"),
        re.compile(r"\bhagb\b"),
    ],
    "ceza_ertelenmesi": [
        re.compile(r"\bcezanin ertelenmesine\b"),
        re.compile(r"\bertele\b"),
    ],
    "cvyok": [
        re.compile(r"\bceza verilmesine yer olmadigi\b"),
    ],
    "dusme": [
        re.compile(r"\bdavanin dusmesi\b"),
        re.compile(r"\bdusmesine\b"),
    ],
    "guvenlik_tedbiri": [
        re.compile(r"\bguvenlik tedbiri\b"),
        re.compile(r"\btck\s*53\b"),
    ],
    "ret": [
        re.compile(r"\btemyiz isteminin reddine\b"),
        re.compile(r"\bistinaf basvurusunun reddine\b"),
    ],
    "bozma": [
        re.compile(r"\bbozulmasina\b"),
        re.compile(r"\bbozularak\b"),
    ],
    "onama": [
        re.compile(r"\bonanmasina\b"),
    ],
}

HEADING_PATTERNS = {
    "gerekce_dusunuldu": re.compile(r"\bgeregi dusunuldu\b"),
    "hukum": re.compile(r"\bhukum\b\s*[:\-]?"),
    "karar": re.compile(r"\bkarar\b\s*[:\-]?"),
}


FALLBACK_DECISION_PATTERNS = [
    re.compile(r"\bsanigin[^\n\.]{0,180}(cezalandirilmasina|beraatine|mahkumiyetine|dusmesine|ceza verilmesine yer olmadigina)"),
    re.compile(r"\bmahkumiyet\b"),
    re.compile(r"\bberaat\b"),
]


def normalize_text(text: str) -> str:
    return text.translate(TRANSLIT_TABLE).lower()


def extract_decision_section(text: str, max_chars: int = 1400) -> Tuple[str, str]:
    norm = normalize_text(text)

    for pattern in SECTION_HEADER_PATTERNS:
        match = pattern.search(norm)
        if match:
            start = match.end()
            section = norm[start : start + max_chars]
            return section.strip(), "header"

    for pattern in FALLBACK_DECISION_PATTERNS:
        match = pattern.search(norm)
        if match:
            start = max(0, match.start() - 160)
            end = min(len(norm), match.end() + 500)
            return norm[start:end].strip(), "fallback"

    return norm[:max_chars].strip(), "prefix"


def classify_categories(decision_section: str) -> List[str]:
    found = []
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(pattern.search(decision_section) for pattern in patterns):
            found.append(category)
    return found or ["unknown"]


def char_word_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    p95_idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.95))
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": float(statistics.median(values)),
        "p95": float(sorted_vals[p95_idx]),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
    }


def sample_indices(total_size: int, sample_size: int, seed: int) -> List[int]:
    if sample_size >= total_size:
        return list(range(total_size))
    rng = random.Random(seed)
    return rng.sample(range(total_size), sample_size)


def audit_dataset(dataset_path: Path, sample_size: int, seed: int, output_dir: Path) -> None:
    ds = load_from_disk(str(dataset_path))["train"]
    total = len(ds)
    indices = sample_indices(total, sample_size, seed)
    sampled = ds.select(indices)

    source_counter = Counter()
    category_counter = Counter()
    extraction_mode_counter = Counter()
    heading_counter = Counter()
    char_lengths = []
    word_lengths = []
    examples_by_category = defaultdict(list)

    row_outputs = []

    for row in sampled:
        text = row.get("text") or ""
        source = row.get("source") or "unknown"
        row_id = row.get("id") or ""

        source_counter[source] += 1
        char_lengths.append(len(text))
        word_lengths.append(len(text.split()))

        normalized_text = normalize_text(text)
        for heading_name, heading_pattern in HEADING_PATTERNS.items():
            if heading_pattern.search(normalized_text):
                heading_counter[heading_name] += 1

        decision_section, mode = extract_decision_section(text)
        extraction_mode_counter[mode] += 1

        categories = classify_categories(decision_section)
        for c in categories:
            category_counter[c] += 1
            if len(examples_by_category[c]) < 3:
                examples_by_category[c].append(decision_section[:320])

        row_outputs.append(
            {
                "id": row_id,
                "source": source,
                "extraction_mode": mode,
                "predicted_categories": "|".join(categories),
                "decision_excerpt": decision_section[:500].replace("\n", " "),
                "text_preview": normalize_text(text)[:200].replace("\n", " "),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_path": str(dataset_path),
        "total_rows": total,
        "sample_size": len(indices),
        "char_length_stats": char_word_stats(char_lengths),
        "word_length_stats": char_word_stats(word_lengths),
        "source_top_20": source_counter.most_common(20),
        "heading_counts": dict(heading_counter),
        "heading_rates": {
            key: round(value / len(indices), 4) if indices else 0.0
            for key, value in heading_counter.items()
        },
        "extraction_mode_counts": dict(extraction_mode_counter),
        "category_counts": dict(category_counter),
        "category_examples": dict(examples_by_category),
    }

    report_path = output_dir / "audit_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "sample_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "source",
                "extraction_mode",
                "predicted_categories",
                "decision_excerpt",
                "text_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(row_outputs)

    print("Audit completed.")
    print(f"Rows scanned: {len(indices)} / {total}")
    print(f"Report: {report_path}")
    print(f"Predictions CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Turkish legal documents and produce silver decision labels."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "saved_dataset",
        help="Path to Hugging Face dataset saved with save_to_disk().",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="How many rows to sample for analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs",
        help="Output directory for report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dataset(
        dataset_path=args.dataset_path,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
