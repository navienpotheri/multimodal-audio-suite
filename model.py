"""
Phase 2: LLaVA-Audio Model Architecture.
Aligns Whisper continuous audio representations with Qwen LLM using a learnable projection layer.
Supports inputs_embeds composition and CPU-optimized frame pooling.
"""

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, AutoTokenizer

class AudioLLM(nn.Module):
    def __init__(self, qwen_model_path, audio_dim=384, pooling_factor=10, torch_dtype=torch.float32):
        super().__init__()
        self.pooling_factor = pooling_factor
        self.torch_dtype = torch_dtype
        
        print(f"Initializing base LLM from: {qwen_model_path}")
        self.llm = Qwen2ForCausalLM.from_pretrained(
            qwen_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        
        # Get LLM hidden dimension
        self.llm_dim = self.llm.config.hidden_size
        print(f"Base LLM Hidden Dimension: {self.llm_dim}")
        
        # Learnable projection layer to map audio encoder dim to LLM dim
        self.proj = nn.Linear(audio_dim, self.llm_dim, dtype=torch_dtype)
        print(f"Audio Projection Layer Initialized: nn.Linear({audio_dim}, {self.llm_dim})")
        
        # Freeze LLM weights (only train projection layer and optionally LoRA)
        self.llm.requires_grad_(False)
        self.proj.requires_grad_(True)
        
    def get_trainable_parameters(self):
        """Returns list of parameters that require gradients."""
        trainable = list(self.proj.parameters())
        # If LoRA is applied to LLM, those parameters will also require grad
        for p in self.llm.parameters():
            if p.requires_grad:
                trainable.append(p)
        return trainable

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    def forward(self, whisper_features, prefix_ids, suffix_ids, labels=None):
        """
        Args:
            whisper_features: [batch_size=1, seq_len=1500, audio_dim]
            prefix_ids: [batch_size=1, prefix_len]
            suffix_ids: [batch_size=1, suffix_len]
            labels: [batch_size=1, label_len] - Target response tokens
        """
        # 1. Project audio features to LLM embedding dimension
        # whisper_features: [1, 1500, audio_dim]
        audio_embeds = self.proj(whisper_features) # [1, 1500, llm_dim]
        
        # 2. Downsample audio sequence using average pooling to speed up CPU processing
        # Pools every k frames (e.g. k=10 reduces 1500 to 150)
        B, L, D = audio_embeds.shape
        k = self.pooling_factor
        if k > 1:
            L_trimmed = L - (L % k)
            audio_embeds = audio_embeds[:, :L_trimmed, :]
            audio_embeds = audio_embeds.view(B, L_trimmed // k, k, D).mean(dim=2) # [1, L//k, llm_dim]
            
        # 3. Look up token embeddings for text prefix and suffix
        prefix_embeds = self.llm.model.embed_tokens(prefix_ids) # [1, prefix_len, llm_dim]
        
        if labels is not None:
            # Training Mode
            suffix_embeds = self.llm.model.embed_tokens(suffix_ids) # [1, suffix_len, llm_dim]
            label_embeds = self.llm.model.embed_tokens(labels) # [1, label_len, llm_dim]
            
            # Concatenate prefix, audio, suffix, and label response
            inputs_embeds = torch.cat([
                prefix_embeds,   # User prompt start
                audio_embeds,    # Projected audio tokens
                suffix_embeds,   # Prompt end/Instruction
                label_embeds     # Target response
            ], dim=1)
            
            # Build target labels for loss mask
            # We set target values to -100 for prefix, audio, and suffix (ignored by cross-entropy)
            p_len = prefix_embeds.shape[1]
            a_len = audio_embeds.shape[1]
            s_len = suffix_embeds.shape[1]
            l_len = label_embeds.shape[1]
            
            targets = torch.full((1, p_len + a_len + s_len + l_len), -100, dtype=torch.long, device=whisper_features.device)
            # The model is trained to predict target labels
            targets[0, -l_len:] = labels[0]
            
            # Run LLM forward pass
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=targets,
                return_dict=True
            )
            return outputs
        else:
            # Inference Mode (Generation)
            # suffix_ids serves as the end of prompt instruction
            suffix_embeds = self.llm.model.embed_tokens(suffix_ids)
            
            # Concatenate prompt prefix, audio, and prompt suffix
            inputs_embeds = torch.cat([
                prefix_embeds,
                audio_embeds,
                suffix_embeds
            ], dim=1)
            
            return inputs_embeds

    def generate(self, whisper_features, prefix_ids, suffix_ids, max_new_tokens=50, temperature=0.7):
        """Generates text autoregressively given audio features and prompt context."""
        self.eval()
        with torch.no_grad():
            # 1. Get initial concatenated embeddings
            inputs_embeds = self.forward(whisper_features, prefix_ids, suffix_ids) # [1, prompt_len, llm_dim]
            
            generated_ids = []
            
            # Autoregressive loop
            for _ in range(max_new_tokens):
                outputs = self.llm(inputs_embeds=inputs_embeds, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature sampling
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                token_id = next_token.item()
                if token_id == self.tokenizer.eos_token_id:
                    break
                    
                generated_ids.append(token_id)
                
                # Get embedding for the newly generated token
                next_embed = self.llm.model.embed_tokens(next_token) # [1, 1, llm_dim]
                
                # Append token embedding to input sequence for next step
                inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
                
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
