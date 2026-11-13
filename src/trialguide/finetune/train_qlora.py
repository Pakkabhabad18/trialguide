"""E1 training (T23): QLoRA on Qwen3-8B with the data from finetune/data.py. Runs on one Kaggle T4.

    pip install -q peft bitsandbytes          # Kaggle already has torch, transformers, accelerate
    python -m trialguide.finetune.train_qlora --data outputs/e1 --out outputs/e1/qlora_r16

Settings (proposal E1): base weights in 4-bit NF4, LoRA rank 16 on every attention and MLP
projection, one epoch. fp16, because the T4 has no bfloat16. Only the answer tokens are trained
on: the prompt (system message, rule and note) is masked out of the loss, so the model learns
what to answer, not to repeat notes.

The prompt is built with the same chat template call the checker uses at test time
(enable_thinking=False), so training and inference see the same text.

Saves the LoRA adapter (about 170 MB) to {out}/adapter, plus the loss log and the settings. If
the Kaggle session stops, rerun the same command: it continues from the last checkpoint.
Use the adapter with the normal checker by setting LLM_HF_4BIT=1 LLM_HF_ADAPTER={out}/adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-8B"
END = "<|im_end|>"          # Qwen's end-of-turn token: the model must learn to stop after the JSON
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(example: dict, tokenizer, max_len: int):
    """Token ids and labels for one example; labels are -100 (ignored) on the prompt part.
    Returns None if the example is longer than max_len (it is skipped, not cut, so the model
    never trains on a note with its end missing)."""
    prompt = tokenizer.apply_chat_template(example["messages"][:2], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    answer = example["messages"][2]["content"] + END
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:
        return None
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}


def collate(batch: list, pad_id: int):
    """Pad a batch on the right; padded positions are ignored by attention and by the loss."""
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
    labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
    mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def main(argv=None) -> None:
    import os
    # one GPU only: with two visible GPUs the Trainer would wrap the 4-bit model in DataParallel
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path("outputs/e1"), help="folder with train/val.jsonl")
    ap.add_argument("--out", type=Path, default=Path("outputs/e1/qlora_r16"))
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch x grad-accum")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1, help="for a quick test run")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load(name):
        rows = read_jsonl(args.data / f"{name}.jsonl")
        rows = rows[: args.limit] if args.limit and name == "train" else rows
        enc = [encode(r, tokenizer, args.max_len) for r in rows]
        kept = [e for e in enc if e is not None]
        print(f"{name}: {len(kept)} examples ({len(rows) - len(kept)} longer than {args.max_len} skipped)")
        return kept

    train, val = load("train"), load("val")

    # 4-bit NF4 base weights (QLoRA), fp16 compute; the whole model on one GPU
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    # keep the trainable LoRA weights in fp32: fp16 training cannot unscale fp16 gradients
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    training = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        fp16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=50, save_total_limit=2,
        report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training, train_dataset=train, eval_dataset=val,
                      data_collator=lambda b: collate(b, tokenizer.pad_token_id))
    checkpoints = list((args.out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(checkpoints))

    model.save_pretrained(args.out / "adapter")
    tokenizer.save_pretrained(args.out / "adapter")
    (args.out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (args.out / "settings.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "quantisation": "4-bit NF4, double quant",
         "lora_alpha": 2 * args.rank, "lora_dropout": 0.05, "target_modules": TARGET_MODULES,
         "n_train": len(train), "n_val": len(val)}, indent=2))
    print("adapter saved to", args.out / "adapter")


if __name__ == "__main__":
    main()
