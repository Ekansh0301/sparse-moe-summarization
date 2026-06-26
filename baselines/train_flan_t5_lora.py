from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import gc

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===========================================================================
# DATA PREPROCESSING
# ===========================================================================

def preprocess_function_t5(examples, tokenizer):
    """Preprocess data for FLAN-T5 with 'summarize:' prefix"""
    inputs = ["summarize: " + doc for doc in examples["document"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=64,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ===========================================================================
# INFERENCE
# ===========================================================================

def generate_summaries_finetuned(model, tokenizer, test_data_raw, device, batch_size=8):
    """Generate summaries with fine-tuned FLAN-T5 model"""
    model.eval()
    predictions = []
    
    print(f"Generating summaries with batch size {batch_size}...")
    
    for i in tqdm(range(0, len(test_data_raw), batch_size), desc="FLAN-T5 Inference"):
        batch = test_data_raw[i:i+batch_size]
        
        inputs = tokenizer(
            batch["document"],
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
    
    return predictions

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """Main execution function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading XSum dataset...")
    dataset = load_dataset("EdinburghNLP/xsum")
    print("✓ Dataset loaded")
    
    # ===========================================================================
    # MODEL LOADING
    # ===========================================================================
    
    encoder_decoder_model_name = "google/flan-t5-base"
    
    print(f"\nLoading model: {encoder_decoder_model_name}")
    ed_tokenizer = T5Tokenizer.from_pretrained(encoder_decoder_model_name)
    ed_model = T5ForConditionalGeneration.from_pretrained(encoder_decoder_model_name)
    
    ed_model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")
    
    ed_model = ed_model.to(device)
    
    print("\n" + "="*80)
    print("CONFIGURING LoRA FOR FLAN-T5")
    print("="*80)
    
    lora_config_t5 = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    ed_model = get_peft_model(ed_model, lora_config_t5)
    ed_model.print_trainable_parameters()
    
    total_params = sum(p.numel() for p in ed_model.parameters())
    trainable_params = sum(p.numel() for p in ed_model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"FLAN-T5 Model Information (with LoRA)")
    print(f"{'='*60}")
    print(f"Model name: {encoder_decoder_model_name}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print(f"Vocabulary size: {len(ed_tokenizer)}")
    print(f"Device: {device}")
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config_t5.r}")
    print(f"  Alpha: {lora_config_t5.lora_alpha}")
    print(f"  Target modules: {lora_config_t5.target_modules}")
    print(f"  Dropout: {lora_config_t5.lora_dropout}")
    print(f"\n✓ FLAN-T5 model with LoRA loaded successfully!")
    
    # ===========================================================================
    # DATA PREPROCESSING
    # ===========================================================================
    
    print("\nPreprocessing datasets...")
    
    train_dataset = dataset["train"].map(
        lambda x: preprocess_function_t5(x, ed_tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train set"
    )
    
    val_dataset = dataset["validation"].map(
        lambda x: preprocess_function_t5(x, ed_tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing validation set"
    )
    
    test_dataset_processed = dataset["test"].map(
        lambda x: preprocess_function_t5(x, ed_tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing test set"
    )
    
    print(f"\n✓ Preprocessing complete!")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset_processed):,}")
    
    # ===========================================================================
    # TRAINING CONFIGURATION
    # ===========================================================================
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan_t5_xsum_lora",
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=2000,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=64,
        fp16=True,
        gradient_checkpointing=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        report_to="none",
        warmup_steps=500,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Output directory: {training_args.output_dir}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Mixed precision (fp16): {training_args.fp16}")
    print("\n✓ Training arguments configured!")
    
    # ===========================================================================
    # METRICS
    # ===========================================================================
    
    rouge_metric = evaluate.load("rouge")
    
    def compute_metrics(eval_pred):
        """Compute ROUGE scores for evaluation"""
        predictions, labels = eval_pred
        
        vocab_size = len(ed_tokenizer)
        predictions = np.clip(predictions, 0, vocab_size - 1)
        
        decoded_preds = ed_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, ed_tokenizer.pad_token_id)
        decoded_labels = ed_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        return {k: round(v * 100, 2) for k, v in result.items()}
    
    print("✓ Compute metrics function defined!")
    
    # ===========================================================================
    # TRAINING
    # ===========================================================================
    
    print("\nClearing GPU memory before training...")
    clear_memory()
    
    print("\n" + "="*80)
    print("INITIALIZING FLAN-T5 TRAINER")
    print("="*80)
    
    data_collator = DataCollatorForSeq2Seq(
        ed_tokenizer, 
        model=ed_model,
        padding=True
    )
    
    trainer = Seq2SeqTrainer(
        model=ed_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=ed_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n✓ Trainer initialized!")
    print(f"Training dataset size: {len(train_dataset):,}")
    print(f"Validation dataset size: {len(val_dataset):,}")
    
    print(f"\n{'='*80}")
    print("STARTING FLAN-T5 LoRA TRAINING")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    ed_model.save_pretrained("./flan_t5_xsum_lora_final")
    ed_tokenizer.save_pretrained("./flan_t5_xsum_lora_final")
    
    print("\n✓ LoRA adapters saved to './flan_t5_xsum_lora_final'")
    print("✓ Training complete!")
    
    # ===========================================================================
    # INFERENCE
    # ===========================================================================
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE WITH FINE-TUNED FLAN-T5")
    print("="*80)
    
    ed_model.to(device)
    
    use_full_test = False
    if use_full_test:
        test_for_inference = dataset["test"]
    else:
        test_for_inference = dataset["test"].select(range(1000))
    
    print(f"\nGenerating summaries for {len(test_for_inference):,} samples...")
    
    ed_predictions = generate_summaries_finetuned(
        ed_model,
        ed_tokenizer,
        test_for_inference,
        device,
        batch_size=8
    )
    
    print(f"\n✓ Generated {len(ed_predictions):,} summaries!")
    
    ed_results_df = pd.DataFrame({
        'document': test_for_inference["document"],
        'reference': test_for_inference["summary"],
        'prediction': ed_predictions
    })
    
    ed_results_df.to_csv('flan_t5_predictions.csv', index=False)
    print(f"✓ Results saved to 'flan_t5_predictions.csv'")
    
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*80}")
    
    for i in range(3):
        print(f"\n{'-'*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'-'*80}")
        print(f"\nDocument (first 200 chars):")
        print(test_for_inference[i]['document'][:200] + "...")
        print(f"\nReference: {test_for_inference[i]['summary']}")
        print(f"\nPrediction: {ed_predictions[i]}")
    
    del ed_model
    clear_memory()
    print("\n✓ FLAN-T5 inference complete and memory cleared!")

if __name__ == "__main__":
    main()
