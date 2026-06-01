from __future__ import annotations

import argparse
import json
from pathlib import Path

from spot_vlm_utils import (
    DEFAULT_PROMPT,
    build_training_example,
    get_image_pair_paths,
    load_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Spot-the-Diff data for Qwen2.5-VL SFT.")
    parser.add_argument("--train_json", default="spot-the-diff/reformat_train.json", help="Training split JSON.")
    parser.add_argument("--val_json", default="spot-the-diff/reformat_val.json", help="Validation split JSON.")
    parser.add_argument("--image_dir", default="spot-the-diff/images", help="Spot image directory.")
    parser.add_argument("--output_dir", default="prepared/qwen25_vl_spot_sft", help="Output directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to pair with each image pair.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit source train pairs.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit source val pairs.")
    return parser.parse_args()


def expand_split(split_data, image_dir: Path, prompt: str, max_samples: int | None):
    if max_samples is not None:
        split_data = split_data[:max_samples]

    rows = []
    for item in split_data:
        image_id = item["img_id"]
        before_path, after_path = get_image_pair_paths(image_dir, image_id)
        for sentence in item["sentences"]:
            rows.append(build_training_example(image_id, before_path, after_path, prompt, sentence))
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    train_rows = expand_split(load_split(Path(args.train_json)), image_dir, args.prompt, args.max_train_samples)
    val_rows = expand_split(load_split(Path(args.val_json)), image_dir, args.prompt, args.max_val_samples)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print(f"Wrote {len(train_rows)} train examples to {train_path}")
    print(f"Wrote {len(val_rows)} val examples to {val_path}")


if __name__ == "__main__":
    main()
