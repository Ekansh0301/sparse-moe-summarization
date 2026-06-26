### Baseline Model Configurations

| Model | Base Checkpoint | Training Method | Learning Rate | Epochs | Batch Size | LoRA Rank (r) | Weight Decay | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BART** | `facebook/bart-large-xsum` | N/A (Pre-finetuned) | N/A | N/A | N/A | N/A | N/A | SOTA Benchmark. |
| **T5-Base** | `google/flan-t5-base` | LoRA (PEFT) | `3e-4` | 3 | 8 | 16 | 0.01 | PEFT on `q` and `v` modules. |
| **Llama-1B** | `unsloth/Llama-3.2-1B` | QLoRA (Unsloth) | `2e-4` | 3 | 16 | 16 | N/A | 4-bit quantization. |

### MoE (From Scratch) Configurations

**Common Parameters:**

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

## Results and Analysis

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


Our analysis of the metrics although expected nonetheless identified some clearer trends. As we anticipated, the pre-finetuned BART model was the notable forerunner, collecting the most favorable ROUGE and BLEU scores overall. The T5-Base model achieved highest rank regarding BERTScore F1, again signifying that its summaries were the closest in semantic similarity to the original summaries, and it had a robust "Extractiveness" score suggesting it tended to copy text more frequently. The Llama3,2-1B was the most consistent in terms of the LARGEST summary length producing the highest "Compression" score of the group. Meanwhile, all of the MoE (mixture-of-experts) models although produced from scratch did far worse against the baselines.. We suspect this may have happened in part due to limited training time but it is exceptionally difficult to train even sparse models without a significant amount of training data . The models were found to have low extractiveness scores, signaling they were generating more "abstractive" (original) summaries; yet these summaries were poor quality.

Our examination of the various metrics, while not necessarily surprising, did show some clearer trends. As expected, the pre-finetuned BART model ended up being the group leader, gaining the best overall ROUGE and BLEU scores. The T5-Base model came in best rank with respect to BERTScore F1, which further indicated that its summaries were most similar in semantic meaning to the original summaries and it had a considerably high "Extractiveness" score suggesting it was copying text more frequently than the other models. The Llama3,2-1B was the model that performed the best regarding THE consistency to the LARGEST summary length, yielding the best "Compression" score overall in the small sample we used. While all MoE (mixture-of-experts) models did not do as well comparatively to the baselines, they were all developed from scratch. We suspect that this might have been due to limitations in training time - which could have biased the results. However, it is very difficult to train even sparse models unless you have a great deal of training data. The models had low extractive scores, suggesting that the summaries they generated were more "abstractive" (original), but with overall poor quality.

MODELS: [https://huggingface.co/collections/Ekansh112/models-anlp]