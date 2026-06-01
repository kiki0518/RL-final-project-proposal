from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, Image as HFImage, Sequence as HFSequence
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen2.5-VL on Spot-the-Diff.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name.")
    parser.add_argument("--train_jsonl", default="prepared/qwen25_vl_spot_sft/train.jsonl", help="Prepared train JSONL.")
    parser.add_argument("--val_jsonl", default="prepared/qwen25_vl_spot_sft/val.jsonl", help="Prepared val JSONL.")
    parser.add_argument("--output_dir", default="outputs/qwen25_vl_spot_qlora", help="Training output directory.")
    parser.add_argument("--attn_implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--min_pixels", type=int, default=None, help="Optional processor min_pixels override.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional processor max_pixels override.")
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override epochs with a fixed max_steps.")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to a trainer checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_double_quant", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_jsonl_dataset(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("images", HFSequence(feature=HFImage()))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_jsonl_dataset(Path(args.train_jsonl))
    eval_dataset = load_jsonl_dataset(Path(args.val_jsonl)) if Path(args.val_jsonl).exists() else None

    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    compute_dtype = resolve_dtype(args.torch_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=not args.disable_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    device_map = None
    if torch.cuda.is_available():
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        attn_implementation=args.attn_implementation,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    gradient_checkpointing = not args.disable_gradient_checkpointing
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=args.save_total_limit,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        max_length=None,
        optim="paged_adamw_8bit",
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        lr_scheduler_type="cosine",
        seed=args.seed,
        eos_token="<|im_end|>",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))
    print(f"Saved adapter and processor to {final_adapter_dir}")


if __name__ == "__main__":
    main()
