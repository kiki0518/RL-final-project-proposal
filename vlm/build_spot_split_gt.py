from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a COCO-style GT file for a Spot split.")
    parser.add_argument("--split_json", required=True, help="Path to reformat_train/val/test.json")
    parser.add_argument("--output_json", required=True, help="Output COCO-style GT path")
    parser.add_argument("--description", default="Spot-the-Diff split evaluation set")
    return parser.parse_args()


def build_coco_gt(items, description: str):
    info = {
        "contributor": "generated",
        "date_created": "generated",
        "description": description,
        "url": "generated",
        "version": "1.0",
        "year": "2026",
    }

    images = []
    annotations = []
    ann_id = 0
    for item in items:
        image_id = f"{item['img_id']}.png"
        images.append({"filename": image_id, "id": image_id})
        for caption in item["sentences"]:
            annotations.append({"caption": caption, "id": ann_id, "image_id": image_id})
            ann_id += 1

    return {
        "info": info,
        "licenses": info,
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    args = parse_args()
    split_path = Path(args.split_json)
    output_path = Path(args.output_json)

    with split_path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)

    coco_gt = build_coco_gt(items, args.description)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(coco_gt, handle, ensure_ascii=False)

    print(f"Wrote {len(coco_gt['images'])} images and {len(coco_gt['annotations'])} captions to {output_path}")


if __name__ == "__main__":
    main()
