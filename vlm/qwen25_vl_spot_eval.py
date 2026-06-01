from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from spot_vlm_utils import DEFAULT_PROMPT, build_inference_messages, get_image_pair_paths, load_split, normalize_text


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot Spot-the-Diff evaluation with Qwen2.5-VL.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Hugging Face model name.")
    parser.add_argument("--data_json", default="spot-the-diff/reformat_test.json", help="Spot split JSON.")
    parser.add_argument("--image_dir", default="spot-the-diff/images", help="Directory containing Spot images.")
    parser.add_argument("--output_json", default="outputs/qwen25_vl_spot/hyp.json", help="Prediction JSON path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for every image pair.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation length.")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--processor_name", default=None, help="Optional processor source.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit.")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", help="Enable nested quantization.")
    parser.add_argument("--bnb_4bit_quant_type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--min_pixels", type=int, default=None, help="Optional processor min_pixels override.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional processor max_pixels override.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation for quick tests.")
    parser.add_argument("--save_every", type=int, default=25, help="Write partial predictions every N examples.")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def generate_caption(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    before_path: Path,
    after_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = build_inference_messages(before_path, after_path, prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return normalize_text(output_text)


def save_predictions(path: Path, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_json)
    image_dir = Path(args.image_dir)
    output_path = Path(args.output_json)

    split_data = load_split(data_path)
    if args.max_samples is not None:
        split_data = split_data[: args.max_samples]

    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor_source = args.processor_name or args.adapter_path or args.model_name
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)

    model_kwargs = {
        "torch_dtype": resolve_dtype(args.torch_dtype),
        "attn_implementation": args.attn_implementation,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=resolve_dtype("bfloat16" if args.torch_dtype == "auto" else args.torch_dtype),
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()

    predictions = []
    for index, item in enumerate(tqdm(split_data, desc="Evaluating")):
        image_id = item["img_id"]
        before_path, after_path = get_image_pair_paths(image_dir, image_id)

        caption = generate_caption(
            model=model,
            processor=processor,
            before_path=before_path,
            after_path=after_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append({"caption": caption, "image_id": f"{image_id}.png"})

        if args.save_every > 0 and (index + 1) % args.save_every == 0:
            save_predictions(output_path, predictions)

    save_predictions(output_path, predictions)
    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
