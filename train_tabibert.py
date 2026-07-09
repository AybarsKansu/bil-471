import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = metric.compute(predictions=predictions, references=labels, average="weighted")
    acc = acc_metric.compute(predictions=predictions, references=labels)
    
    return {
        "f1": f1["f1"],
        "accuracy": acc["accuracy"]
    }

def main():
    model_name = "boun-tabilab/TabiBERT"
    dataset_path = "dataset_finetune_ready"
    output_dir = "./tabibert-finetuned-classification"
    
    print("Dataset yükleniyor...")
    dataset = load_from_disk(dataset_path)
    
    print("Model ve Tokenizer yükleniyor...")
    # Etiket sayısını (num_labels) dataset'in eşsiz etiket sayısına göre al
    unique_labels = len(set(dataset["train"]["label"]))
    print(f"Tespit edilen etiket sayısı: {unique_labels}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=unique_labels,
        ignore_mismatched_sizes=True
    )
    
    # Tokenization fonksiyonu
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    print("Veri tokenize ediliyor...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training argümanları
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs_tabibert",
        fp16=torch.cuda.is_available()  # GPU varsa fp16 aktif et
    )
    
    print("Eğitim başlıyor...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    print("Eğitim tamamlandı, model kaydediliyor...")
    trainer.save_model(output_dir)
    print(f"Model {output_dir} konumuna kaydedildi.")
    
    # Son değerlendirme (test seti)
    print("Test seti üzerinde değerlendirme yapılıyor...")
    eval_results = trainer.evaluate()
    print(f"Değerlendirme sonuçları: {eval_results}")

if __name__ == "__main__":
    main()
