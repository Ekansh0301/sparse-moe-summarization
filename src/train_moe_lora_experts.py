"""
Sparse Mixture of Experts (MoE) Transformer with LoRA Support (Bonus 3)
===========================================================================

This implementation extends the basic MoE Transformer with:
- LoRA (Low-Rank Adaptation) for parameter-efficient expert networks
- Grouped Query Attention (GQA) support
- Multiple routing strategies (Hash, Token Choice)
- Load balancing mechanisms

Bonus 3 Features:
- LoRALinear: Low-rank adaptation layers
- LoRAExpert: LoRA-based expert networks
- SparseMoELayerLoRA: MoE layer using LoRA experts
- Significant parameter reduction while maintaining performance
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

# NOTE: This uses Kaggle secrets - replace with your own token management
hf_token = os.environ.get("HF_TOKEN")

class Config:
    """Global configuration for experiments and model architecture"""
    
    # Experiment Setup - Set which experiments to run
    TRAIN_TOKEN_CHOICE = True          # Train Token Choice router
    TRAIN_HASH = False                 # Train Hash router
    TRAIN_WITH_LOAD_BALANCER = True    # Train with load balancer
    TRAIN_WITHOUT_LOAD_BALANCER = False # Train without load balancer
    USE_CUSTOM_GQA = False              # Use custom GQA (Bonus 2)
    
    # Bonus 3: LoRA Configuration
    USE_LORA_EXPERTS = True            # Use LoRA-based experts (Bonus 3)
    LORA_RANK = 8                      # LoRA rank (r) - typically 4, 8, 16, or 32
    LORA_ALPHA = 16                    # LoRA scaling factor (alpha)
    LORA_DROPOUT = 0.1                 # Dropout for LoRA layers
    
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
    # TODO: Replace 'Ekansh112' with your HuggingFace username
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
        # LoRA versions (Bonus 3)
        'token_choice_lora': 'Ekansh112/sparse-moe-Token-Choice-NoLB-LoRA',
        'hash_lora': 'Ekansh112/sparse-moe-Hash-NoLB-LoRA',
        'token_choice_with_LB_lora': 'Ekansh112/sparse-moe-Token-Choice-LoRA',
        'hash_with_LB_lora': 'Ekansh112/sparse-moe-Hash-LoRA',
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

# ============================================================================
# LORA IMPLEMENTATION (BONUS 3)
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Linear Layer implemented from scratch.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1, 
                 freeze_base=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Base linear layer (can be frozen or trainable)
        self.base_layer = nn.Linear(in_features, out_features, bias=True)
        if freeze_base:
            # Freeze base layer parameters
            for param in self.base_layer.parameters():
                param.requires_grad = False
        
        # LoRA low-rank matrices
        # A: (r × in_features) - initialized with Gaussian (Kaiming)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        
        # B: (out_features × r) - initialized with zeros
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.zeros_(self.lora_B)
        
        # Dropout for LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        # Base output: W @ x (frozen or trainable)
        base_output = self.base_layer(x)
        
        # LoRA output: (B @ A) @ x * scaling
        # x: (batch, seq_len, in_features)
        # A: (rank, in_features)
        # B: (out_features, rank)
        
        # Step 1: x @ A^T -> (batch, seq_len, rank)
        lora_output = F.linear(self.dropout(x), self.lora_A)
class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Linear Layer
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1, 
                 freeze_base=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scaling factor for LoRA path
        
        # Base linear layer (frozen or trainable)
        self.base_layer = nn.Linear(in_features, out_features, bias=True)
        if freeze_base:
            # Freeze base layer parameters (typical LoRA usage)
            for param in self.base_layer.parameters():
                param.requires_grad = False
        
        # LoRA low-rank matrices
        # A: (r × in_features) - initialized with Kaiming for gradient flow
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        
        # B: (out_features × r) - initialized with zeros for stability
        # This ensures LoRA starts as identity (no change initially)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.zeros_(self.lora_B)
        
        # Dropout for regularization in LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        """
        Forward pass combining base and LoRA outputs.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
        
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        # Base output: W @ x (frozen or trainable)
        base_output = self.base_layer(x)
        
        # LoRA output: (B @ A) @ x * scaling
        # Step 1: x @ A^T -> (batch, seq_len, rank)
        lora_output = F.linear(self.dropout(x), self.lora_A)
        
        # Step 2: lora_output @ B^T -> (batch, seq_len, out_features)
        lora_output = F.linear(lora_output, self.lora_B)
        
        # Combine: base + scaled LoRA adaptation
        return base_output + lora_output * self.scaling
    
    def extra_repr(self):
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}'
    
class LoRAExpert(nn.Module):
    """
    LoRA-based Expert Network for MoE (Bonus 3)
    """
    def __init__(self, d_model, d_ff, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        
        # First layer with LoRA: d_model -> d_ff
        self.fc1 = LoRALinear(
            in_features=d_model,
            out_features=d_ff,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_base=True  # Base weights frozen, only train LoRA
        )
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Second layer with LoRA: d_ff -> d_model
        self.fc2 = LoRALinear(
            in_features=d_ff,
            out_features=d_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_base=True  # Base weights frozen, only train LoRA
        )
    
    def forward(self, x):
        """
        Forward pass through LoRA expert.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    def get_lora_parameters(self):
        """Get only the LoRA parameters (for targeted optimization)."""
        lora_params = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A, module.lora_B])
        return lora_params
    
    def get_num_lora_params(self):
        """Count number of trainable LoRA parameters."""
        return sum(p.numel() for p in self.get_lora_parameters())
    
class SparseMoELayerLoRA(nn.Module):
    """
    Sparse MoE Layer with LoRA-based Experts (Bonus 3)
    """
    def __init__(self, d_model, num_experts, d_ff, router_type, top_k=2, 
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Create LoRA-based expert networks
        self.experts = nn.ModuleList([
            LoRAExpert(
                d_model=d_model,
                d_ff=d_ff,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            ) for _ in range(num_experts)
        ])
        
        # Router (same as regular MoE - Hash or Token Choice)
        if router_type == 'hash':
            self.router = HashRouter(num_experts, d_model)
        else:
            self.router = TokenChoiceRouter(num_experts, d_model, top_k)
    
    def forward(self, x, use_load_balancer=True, alpha=0.01):
        """
        Forward pass with sparse routing to LoRA experts.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            use_load_balancer: Whether to compute load balancing loss
            alpha: Weight for load balancing loss
        
        Returns:
            tuple: (output, load_loss, routing_weights)
        """
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
        
        # SPARSE dispatch (same as regular MoE - only process assigned tokens)
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
    
    def get_total_lora_params(self):
        """Get total number of LoRA parameters across all experts."""
        total = 0
        for expert in self.experts:
            total += expert.get_num_lora_params()
        return total
    
    def get_total_params(self):
        """Get total number of parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """Get number of trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# LORA TESTING FUNCTION
# ============================================================================
    
def test_lora_implementation():
    """
    Test function to verify LoRA implementation works correctly.
    """
    print("Testing LoRA Implementation...")
    
    # Test LoRALinear
    lora_layer = LoRALinear(in_features=512, out_features=1024, rank=8, alpha=16)
    x = torch.randn(2, 10, 512)  # (batch, seq_len, in_features)
    output = lora_layer(x)
    print(f"✓ LoRALinear: Input {x.shape} -> Output {output.shape}")
    
    # Test LoRAExpert
    expert = LoRAExpert(d_model=512, d_ff=1024, rank=8)
    output = expert(x)
    lora_params = expert.get_num_lora_params()
    total_params = sum(p.numel() for p in expert.parameters())
    print(f"✓ LoRAExpert: {total_params:,} total params, {lora_params:,} LoRA params")
    print(f"  Parameter reduction: {100*(1-lora_params/total_params):.1f}%")
    
    # Test SparseMoELayerLoRA
    moe_layer = SparseMoELayerLoRA(
        d_model=512, num_experts=4, d_ff=1024, 
        router_type='token_choice', top_k=2,
        lora_rank=8, lora_alpha=16
    )
    output, load_loss, weights = moe_layer(x, use_load_balancer=True)
    trainable = moe_layer.get_trainable_params()
    total = moe_layer.get_total_params()
    print(f"✓ SparseMoELayerLoRA: {total:,} total params, {trainable:,} trainable")
    print(f"  Trainable ratio: {100*trainable/total:.2f}%")
    print(f"All LoRA components working correctly!")
    
# Uncomment to run test:
# test_lora_implementation()

# ============================================================================
# GROUPED QUERY ATTENTION (BONUS 2)
# ============================================================================

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
        
        # Project and reshape
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

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================
    
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
# STANDARD MOE LAYER (Non-LoRA)
# ============================================================================

class SparseMoELayer(nn.Module):
    """
    Standard Sparse MoE layer with full dense experts.
    Used when USE_LORA_EXPERTS = False.
    """
    def __init__(self, d_model, num_experts, d_ff, router_type, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Experts - standard dense feedforward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Router - Hash or Token Choice
        if router_type == 'hash':
            self.router = HashRouter(num_experts, d_model)
        else:
            self.router = TokenChoiceRouter(num_experts, d_model, top_k)
    
    def forward(self, x, use_load_balancer=True, alpha=0.01):
        B, T, D = x.shape
        
        # Route tokens to experts
        weights, indices = self.router(x)
        
        # Load balancing loss
        if use_load_balancer:
            expert_usage = weights.sum(dim=(0, 1)) / (B * T)
            target = 1.0 / self.num_experts
            load_loss = alpha * torch.sum((expert_usage - target) ** 2)
        else:
            load_loss = torch.tensor(0.0, device=x.device)
        
        # SPARSE dispatch - only process assigned tokens
        x_flat = x.view(-1, D)
        weights_flat = weights.view(-1, self.num_experts)
        out_flat = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
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
    """
    Transformer Encoder Layer with MoE feedforward.
    Supports both standard and LoRA-based experts, plus GQA.
    """
    def __init__(self, d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa, use_lora=False):
        super().__init__()
        
        # Choose attention mechanism (GQA or standard)
        if use_gqa:
            self.attn = GroupedQueryAttention(d_model, nhead, Config.GQA_NUM_KV_HEADS, dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Choose MoE type (LoRA or standard)
        if use_lora:
            self.moe = SparseMoELayerLoRA(
                d_model, num_experts, d_ff, router_type, top_k,
                lora_rank=Config.LORA_RANK,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT
            )
        else:
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
    """
    Transformer Decoder Layer with MoE feedforward.
    """
    def __init__(self, d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa, use_lora=False):
        super().__init__()
        
        # Choose attention mechanism (GQA or standard)
        if use_gqa:
            self.self_attn = GroupedQueryAttention(d_model, nhead, Config.GQA_NUM_KV_HEADS, dropout)
            self.cross_attn = GroupedQueryAttention(d_model, nhead, Config.GQA_NUM_KV_HEADS, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Choose MoE type (LoRA or standard)
        if use_lora:
            self.moe = SparseMoELayerLoRA(
                d_model, num_experts, d_ff, router_type, top_k,
                lora_rank=Config.LORA_RANK,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT
            )
        else:
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
    """
    Full encoder-decoder Transformer with Sparse MoE layers.
    """
    def __init__(self, vocab_size, d_model, nhead, num_enc, num_dec, num_experts, 
                 d_ff, router_type, top_k, dropout, max_len, use_gqa, use_lora=False):
        super().__init__()
        self.d_model = d_model
        self.use_lora = use_lora
        
        # Token and position embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            MoEEncoderLayer(d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa, use_lora)
            for _ in range(num_enc)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            MoEDecoderLayer(d_model, nhead, num_experts, d_ff, router_type, top_k, dropout, use_gqa, use_lora)
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
        
        # Decode
        tgt_emb = self.embed(tgt) * np.sqrt(self.d_model)
        tgt_pos = self.pos_embed(torch.arange(tgt.size(1), device=tgt.device))
        tgt_emb = self.dropout(tgt_emb + tgt_pos)
        
        output = tgt_emb
        for layer in self.decoder_layers:
            output, load_loss, _ = layer(output, memory, use_load_balancer=use_load_balancer)
            total_load_loss += load_loss
        
        logits = self.output_proj(output)
        return logits, total_load_loss
    
    def print_param_stats(self):
        """Print parameter statistics (useful for LoRA models)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nModel Parameter Statistics:")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        
        if self.use_lora:
            # Count LoRA parameters specifically
            lora_params = 0
            for module in self.modules():
                if isinstance(module, SparseMoELayerLoRA):
                    lora_params += module.get_trainable_params()
            print(f"   LoRA parameters:      {lora_params:,}")
            print(f"   Parameter reduction:  {100*(1 - trainable_params/total_params):.2f}%")

def load_data(tokenizer):
    """
    Load and prepare XSum dataset for training.
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        test_data: Raw test data for generation
    """
    print("Loading dataset...")
    dataset = load_dataset(Config.DATASET_NAME)
    
    # Subset data for faster experimentation
    train_size = Config.TRAIN_SAMPLES or len(dataset['train'])
    val_size = Config.VAL_SAMPLES or len(dataset['validation'])
    test_size = Config.TEST_SAMPLES or len(dataset['test'])
    
    train_data = dataset['train'].select(range(min(train_size, len(dataset['train']))))
    val_data = dataset['validation'].select(range(min(val_size, len(dataset['validation']))))
    test_data = dataset['test'].select(range(min(test_size, len(dataset['test']))))
    
    def collate(batch):
        """Collate function to batch documents and summaries."""
        docs = [item['document'] for item in batch]
        sums = [item['summary'] for item in batch]
        
        # Tokenize with padding and truncation
        src = tokenizer(docs, padding=True, truncation=True, max_length=Config.MAX_LENGTH, return_tensors='pt')
        tgt = tokenizer(sums, padding=True, truncation=True, max_length=Config.MAX_LENGTH, return_tensors='pt')
        
        return {
            'src_ids': src['input_ids'],
            'src_mask': src['attention_mask'],
            'tgt_ids': tgt['input_ids'],
            'tgt_mask': tgt['attention_mask']
        }
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, collate_fn=collate)
    
    return train_loader, val_loader, test_loader, test_data

def train_model(model, train_loader, val_loader, optimizer, scheduler, name, 
                use_load_balancer, start_epoch=0, best_val_loss=float('inf')):
    """
    Train MoE model with support for checkpoint continuation.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training: {name}")
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
            
            # Decoder input: shift target by 1 position
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits, load_loss = model(src, tgt_input, use_load_balancer)
            
            # Cross-entropy loss (ignore padding tokens)
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
        
        # Save if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, name, epoch, val_loss, is_best=True)
            print(f"✨ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % Config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, name, epoch, val_loss, is_best=False)
    
    return results

def evaluate(model, loader, use_load_balancer):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: MoETransformer model
        loader: Data loader
        use_load_balancer: Whether to use load balancing
    
    Returns:
        Average cross-entropy loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            # Shift target for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits, load_loss = model(src, tgt_input, use_load_balancer)
            
            # Compute loss (ignore padding)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            total_loss += ce_loss.item()
    
    return total_loss / len(loader)

def generate(model, src, max_length=100):
    """
    Generate summary using greedy decoding.
    
    Args:
        model: Trained MoETransformer
        src: Source token IDs (batch)
        max_length: Maximum generation length
    
    Returns:
        Generated token IDs
    """
    model.eval()
    B = src.size(0)
    
    # Start with BOS token
    generated = torch.full((B, 1), tokenizer.bos_token_id or tokenizer.pad_token_id, 
                          dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(src, generated, use_load_balancer=False)
            
            # Greedy selection
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_token == tokenizer.eos_token_id).all():
                break
    
    return generated

def evaluate_with_generation(model, test_loader, test_data, name, use_load_balancer):
    """
    Evaluate model by generating summaries and computing metrics.
    """
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
            
            # Decode to text
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
            'document': documents[i][:200] + '...',  # Truncate for readability
            'reference': references[i],
            'generated': hypotheses[i]
        })
    
    return results

def save_checkpoint(model, name, epoch, val_loss, is_best=False):
    """
    Save model checkpoint to disk.
    """
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
    print(f"💾 Saved checkpoint: {filename}")

def push_to_hub(model, name):
    """
    Push model checkpoint to HuggingFace Hub.
    """
    if not Config.PUSH_TO_HUB:
        return
    
    # WARNING: Replace Config.HF_USERNAME ('Ekansh112') with your own username!
    repo_name = f"{Config.HF_USERNAME}/sparse-moe-{name}"
    try:
        create_repo(repo_name, exist_ok=True)
        api = HfApi()
        
        # Upload best checkpoint
        model_path = f"{Config.CHECKPOINT_DIR}/{name}_best.pt"
        if os.path.exists(model_path):
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"{name}_best.pt",
                repo_id=repo_name
            )
            print(f"☁️ Pushed to Hub: {repo_name}")
    except Exception as e:
        print(f"❌ Failed to push to Hub: {e}")

def load_checkpoint_from_hf(model, repo_id, checkpoint_name):
    """
    Load a checkpoint from Hugging Face Hub.
    """
    print(f"\n🔄 Attempting to load checkpoint from HF Hub...")
    print(f"   Repository: {repo_id}")
    print(f"   Checkpoint: {checkpoint_name}")
    
    try:
        # Download checkpoint from HF Hub
        # The token is already set via login() in main()
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            cache_dir=Config.CHECKPOINT_DIR
        )
        
        print(f"✓ Downloaded checkpoint to: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1  # Continue from next epoch
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✓ Loaded checkpoint successfully!")
        print(f"   Previous epoch: {checkpoint.get('epoch', 0)}")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   Continuing from epoch: {start_epoch}")
        
        return model, start_epoch, best_val_loss
        
    except Exception as e:
        print(f"⚠ Failed to load checkpoint from HF Hub: {e}")
        print(f"   Starting training from scratch...")
        return model, 0, float('inf')
    
def main():
    """
    Main training pipeline 
    """
    global device, tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # WARNING: Kaggle-specific authentication using personal secrets
    # Replace this section if not running on Kaggle
    if Config.PUSH_TO_HUB or Config.CONTINUE_FROM_CHECKPOINT:
        print("\n🔐 Logging into Hugging Face...")
        hf_token = os.environ.get("HF_TOKEN")
        login(token=hf_token)  
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Load XSum dataset
    train_loader, val_loader, test_loader, test_data = load_data(tokenizer)
    
    print(f"\n Dataset loaded:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Define experiments to run
    # Each experiment is a tuple: (router_type, name, use_load_balancer, checkpoint_key)
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
        # Add GQA suffix to name and checkpoint key if using GQA (Bonus 2)
        if Config.USE_CUSTOM_GQA:
            name = f"{name}-GQA"
            checkpoint_key = f"{checkpoint_key}_gqa"
        
        # Add LoRA suffix if using LoRA experts (Bonus 3)
        if Config.USE_LORA_EXPERTS:
            name = f"{name}-LoRA"
            checkpoint_key = f"{checkpoint_key}_lora"
        
        print(f"\n{'='*80}")
        print(f" EXPERIMENT: {name}")
        print(f"  Router Type: {router_type}")
        print(f"   Load Balancer: {use_lb}")
        print(f"   GQA (Bonus 2): {Config.USE_CUSTOM_GQA}")
        print(f"   LoRA (Bonus 3): {Config.USE_LORA_EXPERTS}")
        if Config.USE_LORA_EXPERTS:
            print(f"   LoRA rank: {Config.LORA_RANK}, alpha: {Config.LORA_ALPHA}")
        print(f"{'='*80}")
        
        # Create MoE model with specified configuration
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
            use_gqa=Config.USE_CUSTOM_GQA,
            use_lora=Config.USE_LORA_EXPERTS
        ).to(device)
        
        # Print parameter statistics (especially useful for LoRA models)
        model.print_param_stats()
        
        # Load checkpoint if continuing training from HF Hub
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
        
        # Train model (continue from checkpoint if loaded)
        train_results = train_model(
            model, train_loader, val_loader, optimizer, scheduler, 
            name, use_lb, start_epoch, best_val_loss
        )
        
        # Evaluate model with generation
        eval_results = evaluate_with_generation(model, test_loader, test_data, name, use_lb)
        
        # Push best checkpoint to HuggingFace Hub
        push_to_hub(model, name)
        
        # Store results
        all_results[name] = {
            'train': train_results,
            'eval': eval_results
        }
    
    # Save comprehensive results to text file
    with open('results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPARSE MOE TRANSFORMER - CONTINUED TRAINING RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Model: D_MODEL={Config.D_MODEL}, NHEAD={Config.NHEAD}\n")
        f.write(f"  Experts: {Config.NUM_EXPERTS}, D_FF={Config.D_FF}, TOP_K={Config.TOP_K}\n")
        f.write(f"  Training: {Config.NUM_EPOCHS} additional epochs, LR={Config.LEARNING_RATE}\n")
        f.write(f"  Custom GQA (Bonus 2): {Config.USE_CUSTOM_GQA}\n")
        f.write(f"  LoRA Experts (Bonus 3): {Config.USE_LORA_EXPERTS}\n")
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
    print("Results saved to: results.txt")
    
    # Save full results as JSON for programmatic access
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Detailed results saved to: results.json")

if __name__ == "__main__":
    main()