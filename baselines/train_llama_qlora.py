import os, json, re, time
import unsloth  # noqa: F401
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import nltk
from nltk.corpus import stopwords

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

import wandb
from huggingface_hub import HfApi
from transformers import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

# Workaround for unsloth vision compatibility
try:
    import unsloth.models.vision as _unsloth_vision  # noqa: E402
    if getattr(_unsloth_vision, "torch_compiler_set_stance", None) is not None:
        _unsloth_vision.torch_compiler_set_stance = None
except Exception:
    pass

# ===========================================================================
# CONFIGURATION
# ===========================================================================

HUB_MODEL_ID = "varungupta8976/unsloth-llama1b-xsum-sft"
WANDB_PROJECT = "xsum_unsloth_llama1b"
MODEL_NAME = "unsloth/Llama-3.2-1B"
MAX_INPUT = 1024
MAX_OUTPUT = 64
EPOCHS = 3
LR = 2e-4
MICRO_BATCH = 16
GRAD_ACCUM = 3
LORA_R = 16
SEED = 42
DEV = False
RESUME = True

# ===========================================================================
# ENVIRONMENT SETUP
# ===========================================================================

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable TF32 for RTX 4080
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print("✓ TF32 enabled for faster training on RTX 4080")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "| CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB")

# ===========================================================================
# AUTHENTICATION
# ===========================================================================

print("Logging to W&B (interactive) — you may be prompted.")
wandb.login()
api = HfApi()
try:
    who = api.whoami()
    print("Hugging Face user:", who.get("name", "unknown"))
except Exception as e:
    print("Warning: huggingface_hub whoami failed; make sure you ran huggingface-cli login.", e)

# ===========================================================================
# DATASET LOADING
# ===========================================================================

print("Loading XSum dataset...")
raw = load_dataset("EdinburghNLP/xsum")

def make_instruction(ex):
    """Format example as instruction-response pair"""
    prompt = "Summarize the following article in one sentence:\n\n" + ex["document"]
    text = f"Instruction:\n{prompt}\n\nResponse:\n{ex['summary']}"
    return {"text": text}

print("Formatting dataset as instruction examples...")
ds = raw.map(make_instruction, remove_columns=raw["train"].column_names)

if DEV:
    train_ds = ds["train"].select(range(2000))
    val_ds = ds["validation"].select(range(200))
else:
    train_ds = ds["train"]
    val_ds = ds["validation"]

print("Train size:", len(train_ds), "Val size:", len(val_ds))

# ===========================================================================
# CHECKPOINT RESUMING
# ===========================================================================

output_dir = "./unsloth_llama1b_xsum_ckpts"
last_checkpoint = None

if RESUME and os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        print(f"✓ Resuming from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoint found; starting fresh training")
else:
    print("Starting fresh training (RESUME=False or no output_dir)")

# ===========================================================================
# MODEL LOADING
# ===========================================================================

print("Loading quantized model (QLoRA) from", MODEL_NAME, " — this may take a minute...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_INPUT,
    load_in_4bit = True,
    load_in_8bit = False,
    load_in_16bit = False,
    full_finetuning = False,
)

try:
    model.config.use_cache = False
    print("✓ Disabled KV cache during training")
except Exception:
    pass

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=False,
    max_seq_length=MAX_INPUT,
    random_state=SEED,
)

print("Model loaded. Trainable params summary:")
model.print_trainable_parameters()

print(f"✓ Dataset ready. Train: {len(train_ds)}, Val: {len(val_ds)}")

# ===========================================================================
# TRAINING CONFIGURATION
# ===========================================================================

sft_config = SFTConfig(
    per_device_train_batch_size = MICRO_BATCH,
    per_device_eval_batch_size = MICRO_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate = LR,
    num_train_epochs = EPOCHS,
    max_steps = -1,
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    bf16 = True,
    fp16 = False,
    max_grad_norm = 1.0,
    output_dir = output_dir,
    logging_strategy = "steps",
    logging_steps = 50,
    save_strategy = "steps",
    save_steps = 1000,
    eval_strategy = "steps",
    eval_steps = 1000,
    save_total_limit = 5,
    load_best_model_at_end = False,
    dataset_text_field = "text",
    packing = False,
    max_seq_length = MAX_INPUT,
    dataloader_num_workers = 0,
    dataloader_pin_memory = True,
    push_to_hub = False,
    hub_model_id = HUB_MODEL_ID,
    report_to = ["wandb"],
    run_name = WANDB_PROJECT,
    seed = SEED,
)

trainer = SFTTrainer(
    model = model,
    args = sft_config,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    processing_class = tokenizer,
)

# ===========================================================================
# TRAINING CALLBACKS
# ===========================================================================

class GradientMonitor(TrainerCallback):
    """Monitor gradient norms"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            if grad_norm < 1e-6:
                print(f"⚠️  WARNING at step {state.global_step}: grad_norm={grad_norm:.2e} - vanishing gradients!")
            elif grad_norm > 100:
                print(f"⚠️  WARNING at step {state.global_step}: grad_norm={grad_norm:.2f} - exploding gradients!")

class VRAMMonitor(TrainerCallback):
    """Monitor VRAM usage"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and state.global_step % 500 == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pct = (allocated / total) * 100
            if pct > 95:
                print(f"⚠️  WARNING: VRAM {allocated:.1f}/{total:.1f}GB ({pct:.1f}%) - reduce batch size!")

trainer.add_callback(GradientMonitor())
trainer.add_callback(VRAMMonitor())
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0))

# ===========================================================================
# TRAINING
# ===========================================================================

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: XSum (Train: {len(train_ds)}, Val: {len(val_ds)})")
print(f"LoRA rank: {LORA_R}, alpha: 32, dropout: 0.0")
print(f"Micro batch: {MICRO_BATCH}, Grad accum: {GRAD_ACCUM}")
print(f"Effective batch size: {MICRO_BATCH * GRAD_ACCUM}")
print(f"Learning rate: {LR}, Warmup ratio: 0.03")
print(f"Epochs: {EPOCHS}")
print(f"Steps per epoch: ~{len(train_ds) // (MICRO_BATCH * GRAD_ACCUM)}")
print(f"Total steps: ~{(len(train_ds) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS}")
print(f"Checkpoint dir: {output_dir}")
print(f"Save every: 1000 steps (keeping last 3)")
print(f"Eval every: 1000 steps")
print(f"Push to hub: {HUB_MODEL_ID}")
print("="*80 + "\n")

if last_checkpoint:
    print(f"✓ Resuming from checkpoint: {last_checkpoint}\n")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting fresh training\n")
    trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80)

import glob
checkpoint_dirs = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
print(f"\n✓ Checkpoints saved: {len(checkpoint_dirs)}")
for ckpt in checkpoint_dirs[-3:]:
    print(f"  - {ckpt}")
if len(checkpoint_dirs) > 3:
    print(f"  (Showing last 3 of {len(checkpoint_dirs)} total)")

print("\nSaving final model...")
trainer.save_model(os.path.join(output_dir, "final_model"))

# ===========================================================================
# INFERENCE
# ===========================================================================

print("Running inference on test subset...")
test_ds = raw["test"].select(range(1000)) if not DEV else raw["test"].select(range(200))
preds, refs, docs = [], [], []

gen_model = trainer.model
gen_tokenizer = tokenizer

for ex in tqdm(test_ds, desc="Generating"):
    prompt = "Summarize the following article in one sentence:\n\n" + ex["document"]
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(gen_model.device)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_length=MAX_OUTPUT, num_beams=4, early_stopping=True)
    pred = gen_tokenizer.decode(out[0], skip_special_tokens=True)
    preds.append(pred)
    refs.append(ex["summary"])
    docs.append(ex["document"])

results_df = pd.DataFrame({"document": docs, "reference": refs, "prediction": preds})
results_df.to_csv("unsloth_llama1b_xsum_predictions.csv", index=False)
print("Predictions saved -> unsloth_llama1b_xsum_predictions.csv")

# ===========================================================================
# EVALUATION
# ===========================================================================

print("\n" + "="*80)
print("COMPUTING METRICS")
print("="*80)
print(f"Evaluating {len(preds)} test samples...")

assert len(preds) == len(refs) == len(docs), "Prediction/reference count mismatch"
print(f"✓ Generated {len(preds)} summaries successfully\n")

rouge = evaluate.load("rouge")
sacrebleu = evaluate.load("sacrebleu")
bertscore = evaluate.load("bertscore")
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
rouge_res = {k: round(v*100, 4) for k,v in rouge_res.items()}

bleu_res = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
bleu_score = round(bleu_res["score"], 4)

bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")
bert_f1 = float(np.mean(bert_res["f1"])) * 100

comp_ratios = [len(p.split()) / max(1, len(d.split())) for p,d in zip(preds, docs)]

def extractiveness(pred, doc):
    """Fraction of summary words from document"""
    pwords = [w.lower() for w in re.findall(r"\w+", pred) if w.lower() not in STOPWORDS]
    dwords = set([w.lower() for w in re.findall(r"\w+", doc)])
    if not pwords: return 0.0
    return sum(1 for w in pwords if w in dwords) / len(pwords)

extr = [extractiveness(p, d) for p, d in zip(preds, docs)]

metrics = {
    "ROUGE-1": rouge_res.get("rouge1"),
    "ROUGE-2": rouge_res.get("rouge2"),
    "ROUGE-L": rouge_res.get("rougeL"),
    "BLEU": bleu_score,
    "BERTScore-F1": round(bert_f1, 4),
    "CompressionRatio": round(float(np.mean(comp_ratios)), 6),
    "Extractiveness": round(float(np.mean(extr)), 6),
}

print(json.dumps(metrics, indent=2))
pd.DataFrame([metrics]).to_csv("unsloth_llama1b_xsum_metrics.csv", index=False)
print("\n✓ Metrics saved to: unsloth_llama1b_xsum_metrics.csv")

print("\n" + "="*80)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("="*80)
print(f"✓ Model trained and saved to: {output_dir}")
print(f"✓ Final model saved to: {output_dir}/final_model")
print(f"✓ Metrics computed and saved")
print(f"\nNote: To push to HuggingFace Hub manually:")
print(f"  huggingface-cli upload {HUB_MODEL_ID} {output_dir}")
print("="*80 + "\n")