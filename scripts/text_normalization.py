import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\xa0", " ")
    text = text.replace("**", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    output_column: str = "text_clean",
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise KeyError(f"Missing column: {text_column}")

    result = df.copy()
    text_series = result[text_column].astype("string")
    text_series = text_series.str.replace("\xa0", " ", regex=False)
    text_series = text_series.str.replace("**", "", regex=False)
    text_series = text_series.str.replace(r"\r\n|\r", "\n", regex=True)
    text_series = text_series.str.replace(r"[\t ]+", " ", regex=True)
    text_series = text_series.str.replace(r"\n{3,}", "\n\n", regex=True)
    result[output_column] = text_series.str.strip()
    return result


def _iter_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def process_parquet_path(
    input_path: Path,
    output_path: Path,
    text_column: str = "text",
    output_column: str = "text_clean",
) -> None:
    parquet_files = _iter_parquet_files(input_path)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file)
        cleaned_frame = normalize_dataframe(frame, text_column=text_column, output_column=output_column)
        cleaned_frame.to_parquet(output_path / parquet_file.name, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize legal text columns in parquet files.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet file or directory containing parquet shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for cleaned parquet files.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the raw text column.",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="text_clean",
        help="Name of the cleaned text column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_parquet_path(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        output_column=args.output_column,
    )


if __name__ == "__main__":
    main()