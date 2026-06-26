from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
from tqdm import tqdm
import json
from datasets import load_dataset


model_name = "facebook/bart-large-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
if(device == "cude"):
  print("running on cuda")
model.to(device)
model.eval()


dataset = load_dataset("EdinburghNLP/xsum", revision="main")
print(f"Loaded {len(dataset)} samples for evaluation.")    
rouge = evaluate.load("rouge")

generated_summaries = []
references = []

for example in tqdm(dataset["test"], desc="Evaluating"):
    article = example["document"]
    reference = example["summary"]

    # Tokenize input
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt").to(device)

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            num_beams=4,
            length_penalty=2.0,
            max_length=64,
            min_length=11,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    generated_summaries.append(summary)
    references.append(reference)

results = rouge.compute(predictions=generated_summaries, references=references, use_stemmer=True)

output_file = "rouge_scores.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Final ROUGE Results (full XSum test set) ===")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

print(f"\nFull results saved to {output_file}")