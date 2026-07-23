import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "newmindai/Mecellem-Qwen3-4B-TR"
DEFAULT_DATASET_PATH = REPO_ROOT / "dataset_finetune_ready"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "mecellem-qwen3-4b-lora-classification"
DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.class_weights is None or labels is None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a decoder-only Turkish legal model with LoRA for classification."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=",".join(DEFAULT_LORA_TARGETS))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-class-weighted-loss", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def load_label_metadata(num_labels):
    label_file = REPO_ROOT / "categorization_results.json"
    id2label = {i: f"LABEL_{i}" for i in range(num_labels)}

    if label_file.exists():
        with label_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cluster_names = data.get("cluster_names", {})
        for label_id in range(num_labels):
            id2label[label_id] = cluster_names.get(str(label_id), id2label[label_id])

    label2id = {name: label_id for label_id, name in id2label.items()}
    return id2label, label2id


def prepare_dataset(dataset, args):
    dataset = dataset.shuffle(seed=args.seed)

    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if args.max_eval_samples:
        dataset["test"] = dataset["test"].select(
            range(min(args.max_eval_samples, len(dataset["test"])))
        )

    labels = sorted(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))
    label_map = {old_label: new_label for new_label, old_label in enumerate(labels)}

    if labels != list(range(len(labels))):
        dataset = dataset.map(lambda row: {"label": label_map[row["label"]]})

    return dataset, len(labels)


def build_class_weights(train_labels, num_labels):
    counts = np.bincount(np.array(train_labels), minlength=num_labels)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def main():
    args = parse_args()
    gradient_checkpointing = not args.no_gradient_checkpointing

    print(f"Dataset loading from: {args.dataset_path}")
    dataset = load_from_disk(str(args.dataset_path))
    dataset, num_labels = prepare_dataset(dataset, args)
    id2label, label2id = load_label_metadata(num_labels)

    print(dataset)
    print(f"Num labels: {num_labels}")
    print(f"Labels: {id2label}")

    print(f"Tokenizer loading: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Model loading: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = [
        module.strip()
        for module in args.lora_target_modules.split(",")
        if module.strip()
    ]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    remove_columns = [col for col in dataset["train"].column_names if col != "label"]
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    class_weights = None
    if not args.no_class_weighted_loss:
        class_weights = build_class_weights(tokenized_dataset["train"]["label"], num_labels)
        print(f"Class weights: {class_weights.tolist()}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch",
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=True,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training starts...")
    trainer.train()

    print("Saving LoRA adapter and tokenizer...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("Final evaluation...")
    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == "__main__":
    main()
