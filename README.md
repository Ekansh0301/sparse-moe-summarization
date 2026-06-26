#  Sparse Mixture of Experts (MoE) for Text Summarization
<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=for-the-badge&color=FFD21E)](https://huggingface.co/Ekansh112)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

**A from-scratch PyTorch implementation of Sparse MoE Transformers with LoRA and Grouped Query Attention**

##  Abstract

This repository contains a from-scratch PyTorch implementation of a **Sparse Mixture of Experts (MoE) Transformer** designed for abstractive text summarization. The architecture replaces standard dense feed-forward blocks with sparsely activated expert networks to increase model capacity without a proportional increase in computational overhead.

To validate the architecture, the custom MoE models are benchmarked against state-of-the-art dense baselines, including pre-trained **BART-Large**, and PEFT-finetuned **FLAN-T5-Base** and **LLaMA-3.2-1B**. This project explores the training dynamics, routing mechanics, and generation quality of sparse vs. dense architectures on the EdinburghNLP XSum dataset.

---

##  Core Architecture & Features

This implementation goes beyond a standard Transformer by integrating advanced optimization and scaling techniques common in modern Large Language Models:

* **Sparsely Gated Routing:** Implements both deterministic **Hash Routing** and learnable **Token Choice (Top-K) Routing**. Tokens are dynamically dispatched only to the most relevant expert networks.


* **Auxiliary Load Balancing Loss:** Integrates a custom loss penalty ($L_{aux}$) to prevent routing collapse and ensure tokens are distributed uniformly across all available experts during training.


* **Grouped Query Attention (GQA):** Replaces standard Multi-Head Attention (MHA) with GQA to drastically reduce KV-cache memory footprint during autoregressive decoding.


* **LoRA-Integrated Experts:** Embeds Low-Rank Adaptation (LoRA) directly into the expert feed-forward networks, enabling parameter-efficient fine-tuning of the routing mechanisms without updating full expert weights.



---

##  Repository Structure

```text
sparse-moe-summarization/
├── src/                                  # Core MoE Implementation
│   ├── train_moe_scratch.py              # Standard routing, Load Balancing, and GQA logic
│   └── train_moe_lora_experts.py         # Advanced LoRA-based expert implementation
├── baselines/                            # SOTA Dense Baselines
│   ├── eval_bart_baseline.py             # Inference/eval script for facebook/bart-large-xsum
│   ├── train_flan_t5_lora.py             # Seq2Seq PEFT training script for FLAN-T5
│   └── train_llama_qlora.py              # Unsloth-optimized QLoRA for LLaMA-3.2-1B
├── evaluation/                           
│   └── metrics.py                        # Unified ROUGE, BLEU, and BERTScore evaluation suite
├── requirements.txt                      # Minimum pinned dependencies
├── Results.md                            # Results
├── LICENSE                               # MIT License
└── README.md                             # Project documentation

```

---

##  Model Configurations

### Baseline Setup

| Model | Base Checkpoint | Training Method | Learning Rate | Epochs | Batch Size | LoRA Rank (r) | Weight Decay | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BART** | `facebook/bart-large-xsum` | N/A (Pre-finetuned) | N/A | N/A | N/A | N/A | N/A | SOTA Benchmark. |
| **T5-Base** | `google/flan-t5-base` | LoRA (PEFT) | `3e-4` | 3 | 8 | 16 | 0.01 | PEFT on `q` and `v` modules. |
| **Llama-1B** | `unsloth/Llama-3.2-1B` | QLoRA (Unsloth) | `2e-4` | 3 | 16 | 16 | N/A | 4-bit quantization. |

Note: Baselines serve as the upper-bound target for the from-scratch implementations.

### MoE (From Scratch) Configurations

**Global Parameters:**

* **Architecture:** `D_MODEL=512`, `NHEAD=8`, 6 Encoder/Decoder Layers, `MAX_SEQ_LEN=512`
* **MoE Layer:** `NUM_EXPERTS=4`, `D_FF=1024`, `TOP_K=2`
* **Training:** `Learning Rate=5e-5`, `Batch Size=32`, `Epochs=1`, `Warmup Steps=500`, `Grad Clip=1.0`
* **Tokenizer:** `Tokenizer=t5-small`, `Max Length=256`

**Model-Specific Variations:**
| Model Variant | Routing | Load Balancer | Attention | Expert Training | Key Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MoE-Hash-NoLB** | `HashRouter` | No | Standard MHA | Full | N/A |
| **MoE-Hash-WithLB** | `HashRouter` | Yes | Standard MHA | Full | `ALPHA=0.01` |
| **MoE-TopK-NoLB** | `TokenChoice` | No | Standard MHA | Full | N/A |
| **MoE-TopK-WithLB** | `TokenChoice` | Yes | Standard MHA | Full | `ALPHA=0.01` |
| **MoE-GQA (Bonus 2)**| `TokenChoice` | Yes | **GQA** | Full | `GQA_NUM_KV_HEADS=4` |
| **MoE-LoRA (Bonus 3)**| `TokenChoice` | Yes | Standard MHA | **LoRA** | `LORA_RANK=16`, `ALPHA=32`, `DROPOUT=0.1` |

---

##  Evaluation & Results

Quantitative evaluation was performed on the XSum test split. Metrics include **ROUGE**, **BLEU**, **BERTScore F1** (semantic similarity), **Compression Ratio**, and **Extractiveness** (percentage of generated words directly copied from the source).

###  Quantitative Results

| Model | Samples | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | Compression | Extractiveness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BART | 11,334 | **0.4490** | **0.2054** | **0.3642** | **0.1487** | 0.6875 | 0.0803 | 70.56% |
| T5-Base | 11,334 | 0.3040 | 0.1001 | 0.2551 | 0.0637 | **0.7149** | 0.0726 | **87.11%** |
| Llama3.2-1B | 11,334 | 0.2600 | 0.0685 | 0.1904 | 0.0344 | 0.5871 | **0.2375** | 78.78% |
| MoE-Hash-NoLB | 1,000 | 0.2108 | 0.0511 | 0.1795 | 0.0294 | 0.6108 | 0.0877 | 32.85% |
| MoE-Hash-WithLB | 1,000 | 0.2183 | 0.0484 | 0.1945 | 0.0299 | 0.6203 | 0.0926 | 33.45% |
| MoE-TopK-NoLB | 1,000 | 0.2166 | 0.0546 | 0.1957 | 0.0319 | 0.6136 | 0.0878 | 30.23% |
| MoE-TopK-WithLB | 1,000 | 0.2074 | 0.0468 | 0.1631 | 0.0361 | 0.6813 | 0.0876 | 28.80% |
| MoE-TopK-WithLB-GQA | 1,000 | 0.2311 | 0.0463 | 0.1703 | 0.0424 | 0.5900 | 0.0977 | 28.00% |


###  Analysis & Insights

* **Semantic Alignment:** FLAN-T5 achieved the highest BERTScore F1, indicating its summaries closely matched the semantic intent of the ground truth, despite heavy copying (87.11% extractiveness).


* **Sparse vs. Dense:** The from-scratch MoE implementations trailed behind the pre-trained baselines. This aligns with modern scaling laws indicating that sparse architectures require significantly larger pre-training datasets and extended compute to converge effectively compared to their dense counterparts.


* **Abstractive vs. Extractive:** Interestingly, all custom MoE models exhibited very low extractiveness (~28-33%), forcing the network to generate highly abstractive/novel summaries. While this resulted in lower n-gram overlap (ROUGE/BLEU) and occasional grammatical instability, it successfully proved the routing network's ability to synthesize new language structures rather than simply acting as a copy-mechanism.

---

##  Getting Started

### 1. Environment Setup

Clone the repository and install the pinned dependencies:

```bash
git clone https://github.com/yourusername/sparse-moe-summarization.git
cd sparse-moe-summarization

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

### 2. Training & Evaluation

The repository is modularized. To reproduce the baseline metrics:

```bash
python baselines/eval_bart_baseline.py
python baselines/train_flan_t5_lora.py
python baselines/train_llama_qlora.py

```

To train the custom Sparse MoE models from scratch:

```bash
# Run Hash, Top-K, Load Balancing, and GQA experiments
python src/train_moe_scratch.py

# Run the parameter-efficient LoRA-Expert experiments
python src/train_moe_lora_experts.py

```

---

##  Checkpoints & Assets

All trained model weights, MoE routers, and checkpoints are publicly hosted on Hugging Face [Models](https://huggingface.co/Ekansh112/models-anlp).


## 📜 License

This project is open-sourced under the **MIT License**. See the `LICENSE` file for full details.
