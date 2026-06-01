from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_PROMPT = (
    "Picture 1 is the before image and Picture 2 is the after image. "
    "Write a short factual caption describing all visible differences between the two images. "
    "If nothing changed, write: no visible change. Output only the caption."
)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_split(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_image_pair_paths(image_dir: Path, image_id: str):
    before_path = image_dir / f"{image_id}.png"
    after_path = image_dir / f"{image_id}_2.png"
    return before_path, after_path


def build_inference_messages(before_path: Path, after_path: Path, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before_path.resolve().as_uri()},
                {"type": "image", "image": after_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_training_example(image_id: str, before_path: Path, after_path: Path, prompt: str, caption: str):
    return {
        "image_id": f"{image_id}.png",
        "images": [str(before_path.resolve()), str(after_path.resolve())],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": normalize_text(caption)}],
            }
        ],
    }
