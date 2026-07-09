import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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
    # Türkçe için başarılı bir model seçtik (Llama-3-8B tabanlı)
    model_name = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
    dataset_path = "dataset_finetune_ready"
    output_dir = "./llm-lora-classification"
    
    print("Dataset yükleniyor...")
    dataset = load_from_disk(dataset_path)
    
    # LLM eğitimleri uzun sürebileceğinden ve veri setimiz çok büyük olduğundan,
    # Hızlı bir test için sample yapabilirsiniz (İsteğe bağlı)
    # dataset["train"] = dataset["train"].shuffle(seed=42).select(range(10000))
    # dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    unique_labels = len(set(dataset["train"]["label"]))
    print(f"Tespit edilen etiket sayısı: {unique_labels}")
    
    print("Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LLM Sequence Classification için 4-bit config (GPU belleği için)
    print("4-bit Quantization Config ayarlanıyor...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print("Model yükleniyor...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=unique_labels,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Modeli k-bit eğitimine hazırla
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    print("LoRA yapılandırılıyor...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] # LLaMA mimarisi için genellikle q_proj, v_proj
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenization fonksiyonu
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    print("Veri tokenize ediliyor...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training argümanları
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4, # LoRA için genellikle daha yüksek LR
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs_llm_lora",
        optim="paged_adamw_8bit", # 4-bit eğitim için paged optimizer
        fp16=True
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
    
    print("Eğitim tamamlandı, LoRA adaptörleri kaydediliyor...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model adaptörleri {output_dir} konumuna kaydedildi.")
    
    # Son değerlendirme (test seti)
    print("Test seti üzerinde değerlendirme yapılıyor...")
    eval_results = trainer.evaluate()
    print(f"Değerlendirme sonuçları: {eval_results}")

if __name__ == "__main__":
    main()
