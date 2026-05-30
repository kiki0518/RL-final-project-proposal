#!/usr/bin/env python3
import json
import random
import shutil
from pathlib import Path


SOURCE_DATA_DIR = Path("spot-the-diff")
SOURCE_IMAGE_DIR = SOURCE_DATA_DIR / "images"
TOY_DATA_DIR = Path("toy-spot")
TOY_IMAGE_DIR = TOY_DATA_DIR / "images"
TOY_GT_DIR = Path("toy-gt")
TOY_GT_FILE = TOY_GT_DIR / "spot_total_change_captions_reformat.json"

CANONICAL_CAPTION = "people are gone"
SPLIT_SIZES = {"train": 160, "val": 20, "test": 40}
SEED = 42

SUBJECT_WORDS = ["person", "people", "pedestrian", "pedestrians", "man", "woman", "girl", "boy"]
GONE_WORDS = ["gone", "missing", "no longer there", "not there anymore"]
NOISE_WORDS = ["car", "truck", "van", "vehicle", "bus", "bike", "bicycle", "umbrella", "suv", "dumpster"]


def qualifies(item):
    sentences = item.get("sentences", [])
    if len(sentences) != 1:
        return False

    text = sentences[0].lower()
    if not any(word in text for word in SUBJECT_WORDS):
        return False
    if not any(word in text for word in GONE_WORDS):
        return False
    if any(word in text for word in NOISE_WORDS):
        return False
    return True


def load_split(split):
    path = SOURCE_DATA_DIR / f"reformat_{split}.json"
    with path.open() as f:
        data = json.load(f)
    kept = [item for item in data if qualifies(item)]
    random.Random(SEED).shuffle(kept)
    target_size = SPLIT_SIZES[split]
    if len(kept) < target_size:
        raise ValueError(f"Not enough samples for {split}: need {target_size}, found {len(kept)}")
    return kept[:target_size]


def toy_item(item):
    return {"img_id": item["img_id"], "sentences": [CANONICAL_CAPTION]}


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_image(img_id):
    for suffix in [".png", "_2.png"]:
        src = SOURCE_IMAGE_DIR / f"{img_id}{suffix}"
        dst = TOY_IMAGE_DIR / f"{img_id}{suffix}"
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def write_json(path, data):
    with path.open("w") as f:
        json.dump(data, f)


def build_coco_gt(test_items):
    info = {
        "contributor": "toy-task",
        "date_created": "toy-task",
        "description": "Toy Spot-the-Diff evaluation set",
        "url": "toy-task",
        "version": "1.0",
        "year": "2026",
    }

    images = []
    annotations = []
    ann_id = 0
    for item in test_items:
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


def main():
    ensure_clean_dir(TOY_DATA_DIR)
    TOY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(TOY_GT_DIR)

    selected = {split: [toy_item(item) for item in load_split(split)] for split in SPLIT_SIZES}

    all_ids = {item["img_id"] for split_items in selected.values() for item in split_items}
    for img_id in sorted(all_ids, key=int):
        link_image(img_id)

    for split, items in selected.items():
        write_json(TOY_DATA_DIR / f"reformat_{split}.json", items)

    write_json(TOY_GT_FILE, build_coco_gt(selected["test"]))

    print("Created toy task:")
    for split, items in selected.items():
        print(f"  {split}: {len(items)} examples")
    print(f"  images: {len(all_ids)} image pairs linked into {TOY_IMAGE_DIR}")
    print(f"  gt: {TOY_GT_FILE}")
    print(f"  caption: {CANONICAL_CAPTION}")


if __name__ == "__main__":
    main()
