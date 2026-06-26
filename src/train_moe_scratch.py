"""
Sparse Mixture of Experts (MoE) Transformer for Text Summarization
Implementation with Token Choice and Hash routing strategies
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from huggingface_hub import login, create_repo, HfApi, hf_hub_download
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

hf_token = os.environ.get("HF_TOKEN")

class Config:
    """Global configuration for experiments and model architecture"""
    
    # Experiment Setup - Set which experiments to run
    TRAIN_TOKEN_CHOICE = True          # Train Token Choice router
    TRAIN_HASH = False                 # Train Hash router
    TRAIN_WITH_LOAD_BALANCER = True    # Train with load balancer
    TRAIN_WITHOUT_LOAD_BALANCER = False # Train without load balancer
    USE_CUSTOM_GQA = False              # Use custom GQA (Bonus 2)
    
    # Model Architecture
    D_MODEL = 512                       # Model dimension
    NHEAD = 8                           # Number of attention heads
    NUM_ENCODER_LAYERS = 6              # Encoder layers
    NUM_DECODER_LAYERS = 6              # Decoder layers
    NUM_EXPERTS = 4                     # Number of expert networks
    D_FF = 1024                         # Feedforward dimension
    TOP_K = 2                           # Top-K experts to route to
    DROPOUT = 0.1                       # Dropout probability
    MAX_SEQ_LEN = 512                   # Maximum sequence length
    GQA_NUM_KV_HEADS = 4                # For GQA (must divide NHEAD)
    
    # Training Settings
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 500
    GRAD_CLIP = 1.0
    MAX_LENGTH = 256                    # Max tokenization length
    
    # Dataset
    DATASET_NAME = "EdinburghNLP/xsum"
    TRAIN_SAMPLES = 100000              # Set to None for full dataset
    VAL_SAMPLES = 2000
    TEST_SAMPLES = 2000
    
    # Tokenizer
    TOKENIZER_NAME = "t5-small"

    # Checkpoint Continuation
    CONTINUE_FROM_CHECKPOINT = True     # Set to True to load from HF
    CHECKPOINT_REPO_MAPPING = {
        'token_choice': 'Ekansh112/sparse-moe-Token-Choice-NoLB',
        'hash': 'Ekansh112/sparse-moe-Hash-NoLB',
        'token_choice_with_LB': 'Ekansh112/sparse-moe-Token-Choice',
        'hash_with_LB': 'Ekansh112/sparse-moe-Hash',
        # GQA versions
        'token_choice_gqa': 'Ekansh112/sparse-moe-Token-Choice-NoLB-GQA',
        'hash_gqa': 'Ekansh112/sparse-moe-Hash-NoLB-GQA',
        'token_choice_with_LB_gqa': 'Ekansh112/sparse-moe-Token-Choice-GQA',
        'hash_with_LB_gqa': 'Ekansh112/sparse-moe-Hash-GQA',
    }
    
    # Load Balancing
    LOAD_BALANCE_ALPHA = 0.01           # Weight for load balancing loss
    
    # Checkpointing
    CHECKPOINT_DIR = "./checkpoints"
    SAVE_EVERY_N_EPOCHS = 1
    
    # Hugging Face
    HF_USERNAME = "Ekansh112"                # TODO: Replace with your username
    PUSH_TO_HUB = True 
    
    # Evaluation
    EVAL_SAMPLES = 500                  # Number of samples for generation eval
    GENERATION_MAX_LENGTH = 100

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class GroupedQueryAttention(nn.Module):
    """
    Custom Grouped Query Attention (GQA) implementation.
    GQA reduces memory by sharing key/value heads across multiple query heads.
    """
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Q projection: full dimension
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # K, V projections: reduced dimension (shared across query groups)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, T, _ = query.shape
        
        # Project and reshape: [B, T, num_heads, head_dim]
        Q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K,V heads to match query heads (GQA mechanism)
        if self.num_queries_per_kv > 1:
            K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
            V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        
        return out, attn.mean(dim=1)
    
def load_checkpoint_from_hf(model, repo_id, checkpoint_name="Token-Choice-NoLB_best.pt"):
    """
    Load model checkpoint from Hugging Face Hub.
    Useful for continuing training or transfer learning.
    
    Args:
        model: The model to load weights into
        repo_id: HuggingFace repository ID (e.g., 'username/model-name')
        checkpoint_name: Name of the checkpoint file
    
    Returns:
        tuple: (model, start_epoch, best_val_loss)
    """
    try:
        print(f"\nAttempting to load checkpoint from {repo_id}...")
        
        # Download the checkpoint file from HF Hub
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            cache_dir=Config.CHECKPOINT_DIR
        )
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"✓ Previous validation loss: {checkpoint['val_loss']:.4f}")
        
        return model, checkpoint['epoch'], checkpoint['val_loss']
        
    except Exception as e:
        print(f" Failed to load checkpoint from HF: {e}")
        print(f"   Starting training from scratch...")
        return model, 0, float('inf')

# ============================================================================
# ROUTING MECHANISMS
# ============================================================================

class HashRouter(nn.Module):
    """
    Hash-based routing mechanism.
    Uses a fixed hash matrix to deterministically assign tokens to experts.
    Non-learnable routing strategy.
    """
    def __init__(self, num_experts, d_model):
        super().__init__()
        self.num_experts = num_experts
        # Fixed hash matrix (non-trainable)
        self.hash_matrix = nn.Parameter(torch.randn(d_model, num_experts), requires_grad=False)
        
    def forward(self, x):
        # Compute hash scores and select expert with max score
        scores = torch.matmul(x, self.hash_matrix)
        indices = torch.argmax(scores, dim=-1)
        # One-hot encoding for hard routing
        weights = F.one_hot(indices, self.num_experts).float()
        return weights, indices

class TokenChoiceRouter(nn.Module):
    """
    Learnable token-choice Top-K routing.
    Each token learns to select its top-K experts via a gating network.
    """
    def __init__(self, num_experts, d_model, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Learnable gating network
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        # Compute gating scores
        logits = self.gate(x)
        # Select top-K experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        # Softmax over top-K for weighted combination
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Scatter weights into full expert dimension
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, top_k_indices, top_k_weights)
        return weights, top_k_indices

# ============================================================================
# SPARSE MOE LAYER
# ============================================================================

class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer with TRUE SPARSE dispatch.
    Only processes tokens for experts with non-zero weights.
    """
    def __init__(self, d_model, num_experts, d_ff, router_type, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Create expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Initialize router based on type
        if router_type == 'hash':
            self.router = HashRouter(num_experts, d_model)
        else:  # token_choice
            self.router = TokenChoiceRouter(num_experts, d_model, top_k)
    
    def forward(self, x, use_load_balancer=True, alpha=0.01):
        B, T, D = x.shape
        
        # Route tokens to experts
        weights, indices = self.router(x)
        
        # Load balancing loss (encourages uniform expert usage)
        if use_load_balancer:
            expert_usage = weights.sum(dim=(0, 1)) / (B * T)
            target = 1.0 / self.num_experts
            load_loss = alpha * torch.sum((expert_usage - target) ** 2)
        else:
            load_loss = torch.tensor(0.0, device=x.device)
        
        # SPARSE dispatch: only process tokens assigned to each expert
        x_flat = x.view(-1, D)
        weights_flat = weights.view(-1, self.num_experts)
        out_flat = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Only process tokens with non-zero weight for this expert
            mask = weights_flat[:, expert_idx] > 0
            if mask.any():
                tokens = x_flat[mask]
                expert_out = self.experts[expert_idx](tokens)
                expert_weights = weights_flat[mask, expert_idx:expert_idx+1]
                out_flat[mask] += expert_out * expert_weights
        
        output = out_flat.view(B, T, D)
        return output, load_loss, weights

# ============================================================================
# TRANSFORMER ENCODER/DECODER LAYERS
# ============================================================================
    
class MoEEncoderLayer(nn.Module):
    """Transformer encoder layer with MoE feedforward."""
    def __init__(self, d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa):
        super().__init__()
        # Choose between GQA or standard multi-head attention
        if use_gqa:
            self.attn = GroupedQueryAttention(d_model, nhead, Config.GQA_NUM_KV_HEADS, dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # MoE feedforward layer
        self.moe = SparseMoELayer(d_model, num_experts, d_ff, router_type, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, use_load_balancer=True):
        # Self-attention with residual connection
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, key_padding_mask=None)
        x = self.norm1(x + self.dropout(attn_out))
        
        # MoE feedforward with residual connection
        moe_out, load_loss, weights = self.moe(x, use_load_balancer, Config.LOAD_BALANCE_ALPHA)
        x = self.norm2(x + self.dropout(moe_out))
        
        return x, load_loss, weights

class MoEDecoderLayer(nn.Module):
    """Transformer decoder layer with MoE feedforward."""
    def __init__(self, d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa):
        super().__init__()
        # Choose between GQA or standard multi-head attention
        if use_gqa:
            self.self_attn = GroupedQueryAttention(d_model, nhead, Config.GQA_NUM_KV_HEADS, dropout)
            self.cross_attn = GroupedQueryAttention(d_model, nhead, Config.GQA_NUM_KV_HEADS, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # MoE feedforward layer
        self.moe = SparseMoELayer(d_model, num_experts, d_ff, router_type, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None, use_load_balancer=True):
        # Self-attention with residual
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=None)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention with residual
        cross_attn_out, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=None)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # MoE feedforward with residual
        moe_out, load_loss, weights = self.moe(x, use_load_balancer, Config.LOAD_BALANCE_ALPHA)
        x = self.norm3(x + self.dropout(moe_out))
        
        return x, load_loss, weights

# ============================================================================
# FULL MOE TRANSFORMER MODEL
# ============================================================================

class MoETransformer(nn.Module):
    """Full encoder-decoder Transformer with Sparse MoE layers."""
    def __init__(self, vocab_size, d_model, nhead, num_enc, num_dec, num_experts, 
                 d_ff, router_type, top_k, dropout, max_len, use_gqa):
        super().__init__()
        self.d_model = d_model
        
        # Token and position embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            MoEEncoderLayer(d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa)
            for _ in range(num_enc)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            MoEDecoderLayer(d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa)
            for _ in range(num_dec)
        ])
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, use_load_balancer=True):
        # Encode source sequence
        src_emb = self.embed(src) * np.sqrt(self.d_model)
        src_pos = self.pos_embed(torch.arange(src.size(1), device=src.device))
        src_emb = self.dropout(src_emb + src_pos)
        
        memory = src_emb
        total_load_loss = 0
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            memory, load_loss, _ = layer(memory, use_load_balancer=use_load_balancer)
            total_load_loss += load_loss
        
        # Decode target sequence
        tgt_emb = self.embed(tgt) * np.sqrt(self.d_model)
        tgt_pos = self.pos_embed(torch.arange(tgt.size(1), device=tgt.device))
        tgt_emb = self.dropout(tgt_emb + tgt_pos)
        
        output = tgt_emb
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            output, load_loss, _ = layer(output, memory, use_load_balancer=use_load_balancer)
            total_load_loss += load_loss
        
        # Project to vocabulary
        logits = self.output_proj(output)
        return logits, total_load_loss

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(tokenizer):
    """Load and preprocess XSum dataset."""
    print(" Loading dataset...")
    dataset = load_dataset(Config.DATASET_NAME)
    
    # Subset the data according to config
    train_size = Config.TRAIN_SAMPLES or len(dataset['train'])
    val_size = Config.VAL_SAMPLES or len(dataset['validation'])
    test_size = Config.TEST_SAMPLES or len(dataset['test'])
    
    train_data = dataset['train'].select(range(min(train_size, len(dataset['train']))))
    val_data = dataset['validation'].select(range(min(val_size, len(dataset['validation']))))
    test_data = dataset['test'].select(range(min(test_size, len(dataset['test']))))
    
    def collate(batch):
        """Collate function to tokenize batches."""
        docs = [item['document'] for item in batch]
        sums = [item['summary'] for item in batch]
        
        # Tokenize source (documents) and target (summaries)
        src = tokenizer(docs, padding=True, truncation=True, max_length=Config.MAX_LENGTH, return_tensors='pt')
        tgt = tokenizer(sums, padding=True, truncation=True, max_length=Config.MAX_LENGTH, return_tensors='pt')
        
        return {
            'src_ids': src['input_ids'],
            'src_mask': src['attention_mask'],
            'tgt_ids': tgt['input_ids'],
            'tgt_mask': tgt['attention_mask']
        }
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, collate_fn=collate)
    
    return train_loader, val_loader, test_loader, test_data

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler, name, 
                use_load_balancer, start_epoch=0, best_val_loss=float('inf')):
    """
    Training function with support for checkpoint continuation.
    
    Args:
        model: MoE model to train
        train_loader, val_loader: Data loaders
        optimizer, scheduler: Optimization components
        name: Experiment name for saving
        use_load_balancer: Whether to use load balancing loss
        start_epoch: Starting epoch (for checkpoint resuming)
        best_val_loss: Best validation loss so far (for checkpoint resuming)
    """
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"   Load Balancer: {use_load_balancer}")
    print(f"   Starting from epoch: {start_epoch}")
    print(f"   Best previous val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    results = {'train_losses': [], 'val_losses': [], 'start_epoch': start_epoch}
    
    for epoch in range(start_epoch, start_epoch + Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + Config.NUM_EPOCHS}")
        for batch in pbar:
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            # Prepare decoder input/output (shift by one for autoregressive training)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits, load_loss = model(src, tgt_input, use_load_balancer)
            
            # Compute cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Total loss = CE loss + load balancing loss
            loss = ce_loss + load_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            total_loss += ce_loss.item()
            pbar.set_postfix({'loss': ce_loss.item()})
        
        # Evaluate on validation set
        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, use_load_balancer)
        
        results['train_losses'].append(avg_train_loss)
        results['val_losses'].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, name, epoch, val_loss, is_best=True)
            print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % Config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, name, epoch, val_loss, is_best=False)
    
    return results

def evaluate(model, loader, use_load_balancer):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits, load_loss = model(src, tgt_input, use_load_balancer)
            
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            total_loss += ce_loss.item()
    
    return total_loss / len(loader)

# ============================================================================
# GENERATION AND EVALUATION
# ============================================================================

def generate(model, src, max_length=100):
    """Generate summaries using greedy decoding."""
    model.eval()
    B = src.size(0)
    
    # Initialize with BOS token
    generated = torch.full((B, 1), tokenizer.bos_token_id or tokenizer.pad_token_id, 
                          dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(src, generated, use_load_balancer=False)
            # Greedy decoding: take argmax
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_token == tokenizer.eos_token_id).all():
                break
    
    return generated

def evaluate_with_generation(model, test_loader, test_data, name, use_load_balancer):
    """Evaluate model by generating summaries and computing basic metrics."""
    print(f"\n Evaluating: {name}")
    
    model.eval()
    references = []
    hypotheses = []
    documents = []
    
    num_samples = min(Config.EVAL_SAMPLES, len(test_data))
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if len(references) >= num_samples:
                break
            
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            # Generate summaries
            generated = generate(model, src, Config.GENERATION_MAX_LENGTH)
            
            # Decode and collect
            for j in range(src.size(0)):
                if len(references) >= num_samples:
                    break
                
                doc = tokenizer.decode(src[j], skip_special_tokens=True)
                ref = tokenizer.decode(tgt[j], skip_special_tokens=True)
                hyp = tokenizer.decode(generated[j], skip_special_tokens=True)
                
                documents.append(doc)
                references.append(ref)
                hypotheses.append(hyp)
    
    # Compute basic metrics (install rouge_score and bert_score for full metrics)
    results = {
        'model': name,
        'num_samples': len(references),
        'avg_ref_length': np.mean([len(r.split()) for r in references]),
        'avg_hyp_length': np.mean([len(h.split()) for h in hypotheses]),
        'samples': []
    }
    
    # Save first 5 sample generations
    for i in range(min(5, len(references))):
        results['samples'].append({
            'document': documents[i][:200] + '...',
            'reference': references[i],
            'generated': hypotheses[i]
        })
    
    return results

# ============================================================================
# CHECKPOINTING AND HUB UPLOAD
# ============================================================================

def save_checkpoint(model, name, epoch, val_loss, is_best=False):
    """Save model checkpoint to disk."""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'config': {
            'D_MODEL': Config.D_MODEL,
            'NUM_EXPERTS': Config.NUM_EXPERTS,
            'name': name
        }
    }
    
    filename = f"{name}_best.pt" if is_best else f"{name}_epoch{epoch}.pt"
    path = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, path)
    print(f" Saved checkpoint: {filename}")

def push_to_hub(model, name):
    """Upload model checkpoint to HuggingFace Hub."""
    if not Config.PUSH_TO_HUB:
        return
    
    
    repo_name = f"{Config.HF_USERNAME}/sparse-moe-{name}"
    try:
        create_repo(repo_name, exist_ok=True)
        api = HfApi()
        
        # Upload model checkpoint
        model_path = f"{Config.CHECKPOINT_DIR}/{name}_best.pt"
        if os.path.exists(model_path):
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"{name}_best.pt",
                repo_id=repo_name
            )
            print(f"  Pushed to Hub: {repo_name}")
    except Exception as e:
        print(f"  Failed to push to Hub: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function to run all experiments."""
    global device, tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Login to HuggingFace (for pushing models or loading checkpoints)
    if Config.PUSH_TO_HUB or Config.CONTINUE_FROM_CHECKPOINT:
        print("\n Logging into Hugging Face...")
        # NOTE: Replace with your own token management
        hf_token = os.environ.get("HF_TOKEN") 
        login(token=hf_token)  
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Load dataset
    train_loader, val_loader, test_loader, test_data = load_data(tokenizer)
    
    print(f"\nDataset loaded:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Define experiments to run
    experiments = []
    
    # Task 2.1: Token Choice and Hash routers with Load Balancer
    if Config.TRAIN_TOKEN_CHOICE and Config.TRAIN_WITH_LOAD_BALANCER:
        experiments.append(('token_choice', 'Token-Choice', True, 'token_choice_with_LB'))
    if Config.TRAIN_HASH and Config.TRAIN_WITH_LOAD_BALANCER:
        experiments.append(('hash', 'Hash', True, 'hash_with_LB'))
    
    # Bonus 1: Without load balancer
    if Config.TRAIN_WITHOUT_LOAD_BALANCER:
        if Config.TRAIN_TOKEN_CHOICE:
            experiments.append(('token_choice', 'Token-Choice-NoLB', False, 'token_choice'))
        if Config.TRAIN_HASH:
            experiments.append(('hash', 'Hash-NoLB', False, 'hash'))
    
    all_results = {}
    
    # Run each experiment
    for router_type, name, use_lb, checkpoint_key in experiments:
        # Add GQA suffix if using custom GQA
        if Config.USE_CUSTOM_GQA:
            name = f"{name}-GQA"
            checkpoint_key = f"{checkpoint_key}_gqa"
        
        print(f"\n{'='*80}")
        print(f" EXPERIMENT: {name} (Load Balancer: {use_lb}, GQA: {Config.USE_CUSTOM_GQA})")
        print(f"{'='*80}")
        
        # Create model
        model = MoETransformer(
            vocab_size=vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_enc=Config.NUM_ENCODER_LAYERS,
            num_dec=Config.NUM_DECODER_LAYERS,
            num_experts=Config.NUM_EXPERTS,
            d_ff=Config.D_FF,
            router_type=router_type,
            top_k=Config.TOP_K,
            dropout=Config.DROPOUT,
            max_len=Config.MAX_SEQ_LEN,
            use_gqa=Config.USE_CUSTOM_GQA
        ).to(device)
        
        print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load checkpoint if continuing training
        start_epoch = 0
        best_val_loss = float('inf')
        
        if Config.CONTINUE_FROM_CHECKPOINT and checkpoint_key in Config.CHECKPOINT_REPO_MAPPING:
            repo_id = Config.CHECKPOINT_REPO_MAPPING[checkpoint_key]
            checkpoint_name = f"{name}_best.pt"
            model, start_epoch, best_val_loss = load_checkpoint_from_hf(
                model, repo_id, checkpoint_name
            )
        
        # Create optimizer and scheduler (recreate for continued training)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        total_steps = len(train_loader) * Config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, Config.WARMUP_STEPS, total_steps)
        
        # Train model (continues from checkpoint if loaded)
        train_results = train_model(
            model, train_loader, val_loader, optimizer, scheduler, 
            name, use_lb, start_epoch, best_val_loss
        )
        
        # Evaluate with generation
        eval_results = evaluate_with_generation(model, test_loader, test_data, name, use_lb)
        
        # Push to HuggingFace Hub
        push_to_hub(model, name)
        
        # Store results
        all_results[name] = {
            'train': train_results,
            'eval': eval_results
        }
    
    # Save results to text file
    with open('results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPARSE MOE TRANSFORMER - TRAINING RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Model: D_MODEL={Config.D_MODEL}, NHEAD={Config.NHEAD}\n")
        f.write(f"  Experts: {Config.NUM_EXPERTS}, D_FF={Config.D_FF}, TOP_K={Config.TOP_K}\n")
        f.write(f"  Training: {Config.NUM_EPOCHS} epochs, LR={Config.LEARNING_RATE}\n")
        f.write(f"  Custom GQA (Bonus 2): {Config.USE_CUSTOM_GQA}\n")
        f.write(f"  Continued from checkpoint: {Config.CONTINUE_FROM_CHECKPOINT}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        for name, results in all_results.items():
            f.write(f"\nMODEL: {name}\n")
            f.write("-"*80 + "\n")
            f.write(f"Started from epoch: {results['train']['start_epoch']}\n")
            f.write(f"Final Train Loss: {results['train']['train_losses'][-1]:.4f}\n")
            f.write(f"Final Val Loss: {results['train']['val_losses'][-1]:.4f}\n")
            f.write(f"Avg Reference Length: {results['eval']['avg_ref_length']:.1f} words\n")
            f.write(f"Avg Generated Length: {results['eval']['avg_hyp_length']:.1f} words\n")
            f.write("\nSample Generations:\n")
            
            for i, sample in enumerate(results['eval']['samples'], 1):
                f.write(f"\n  Sample {i}:\n")
                f.write(f"    Document: {sample['document']}\n")
                f.write(f"    Reference: {sample['reference']}\n")
                f.write(f"    Generated: {sample['generated']}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    print("\nAll experiments completed!")
    print(" Results saved to: results.txt")
    
    # Save full results as JSON
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(" Detailed results saved to: results.json")

if __name__ == "__main__":
    main()