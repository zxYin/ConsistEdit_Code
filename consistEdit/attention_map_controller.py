import os
import torch
import math
from .global_var import GlobalVars

class AttentionMapController:
    def __init__(self, save_dir, thres=0.1, layer_num=24):
        self.save_dir = save_dir
        self.thres = thres
        self.cross_attn_sum = None
        self.cross_attn_count = 0
        self.reverse_cross_attn_sum = None
        self.reverse_cross_attn_count = 0
        self.video_self_attn_sum = None
        self.video_self_attn_count = 0
        self.self_attn_mask = None
        self.text_self_attn_sum = None
        self.text_self_attn_count = 0

        self.layer_cross_attn_sum = [None] * layer_num
        self.layer_cross_attn_count = [0] * layer_num
        print(f"Set mask save directory for AttentionMapController: {save_dir}")

    def get_self_attn_mask(self):
        return self.self_attn_mask
    def get_cross_attn_sum(self):
        return self.cross_attn_sum
    def get_reverse_cross_attn_sum(self):
        return self.reverse_cross_attn_sum
    def get_cross_attn_count(self):
        return self.cross_attn_count
    
    def set_self_attn_mask(self, mask):
        self.self_attn_mask = mask

    def reset_cross_attns(self):
        self.cross_attn_sum = None
        self.cross_attn_count = 0
    def reset_reverse_cross_attns(self):
        self.reverse_cross_attn_sum = None
        self.reverse_cross_attn_count = 0
    def reset_video_self_attn(self):
        self.video_self_attn_sum = None
        self.video_self_attn_count = 0
    def reset_text_self_attn(self):
        self.text_self_attn_sum = None
        self.text_self_attn_count = 0
    
    def reset_self_attn_mask(self):
        self.self_attn_mask = None

    def save_cur_attn_map(self, q, k, cur_step, layer_id):
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize result tensor (on GPU)
        attn_map_mean = torch.zeros(batch_size, seq_len, seq_len, device=q.device, dtype=q.dtype)
        
        # Parameters for batch computation
        batch_chunk = 1  # Number of batches to process at a time
        head_chunk = 1   # Number of attention heads to process at a time, can be adjusted based on GPU memory
        for i in range(0, batch_size, batch_chunk):
            for j in range(0, num_heads, head_chunk):
                # Select data for current batch and attention heads
                q_chunk = q[i:i+batch_chunk, j:j+head_chunk]
                k_chunk = k[i:i+batch_chunk, j:j+head_chunk]
                
                # Calculate attention scores for current attention heads
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                
                # Add to mean attention map
                attn_map_mean[i:i+batch_chunk] += attn_probs.sum(dim=1)
        
        # Calculate mean
        attn_map_mean /= num_heads
        video_to_text_attn = attn_map_mean[:, :-GlobalVars.TEXT_LENGTH, -GlobalVars.TEXT_LENGTH:]

        # Update accumulated sum and count
        if self.cross_attn_sum is None:
            self.cross_attn_sum = video_to_text_attn
        else:
            self.cross_attn_sum += video_to_text_attn
        self.cross_attn_count += 1
        
        # Process video self attention
        video_to_video_attn = attn_map_mean[:, :-GlobalVars.TEXT_LENGTH:, :-GlobalVars.TEXT_LENGTH]
        if self.video_self_attn_sum is None:
            self.video_self_attn_sum = video_to_video_attn
        else:
            self.video_self_attn_sum += video_to_video_attn
        self.video_self_attn_count += 1

        text_to_text_attn = attn_map_mean[:, -GlobalVars.TEXT_LENGTH:, -GlobalVars.TEXT_LENGTH:]
        if self.text_self_attn_sum is None:
            self.text_self_attn_sum = text_to_text_attn
        else:
            self.text_self_attn_sum += text_to_text_attn
        self.text_self_attn_count += 1

    def get_video_value_to_text_mask(self, q, k, layer_idx=None, token_idx=[1]):
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize result tensor (on GPU)
        # Only compute video-to-text attention scores
        attn_map_mean = torch.zeros(batch_size, seq_len - GlobalVars.TEXT_LENGTH, GlobalVars.TEXT_LENGTH, device=q.device, dtype=q.dtype)
        
        # Parameters for batch computation
        batch_chunk = 1  # Number of batches to process at a time
        head_chunk = 2   # Number of attention heads to process at a time, can be adjusted based on GPU memory
        for i in range(0, batch_size, batch_chunk):
            for j in range(0, num_heads, head_chunk):
                # Select only video queries and text keys
                q_chunk = q[i:i+batch_chunk, j:j+head_chunk, GlobalVars.TEXT_LENGTH:]  # Video queries
                k_chunk = k[i:i+batch_chunk, j:j+head_chunk, :GlobalVars.TEXT_LENGTH]  # Text keys
                
                # Calculate attention scores for current attention heads
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                
                # Add to mean attention map
                attn_map_mean[i:i+batch_chunk] += attn_probs.sum(dim=1)
        
        # Calculate mean
        attn_map_mean /= num_heads
        video_to_text_attn = attn_map_mean

        if layer_idx is None:
            if self.cross_attn_sum is None:
                self.cross_attn_sum = video_to_text_attn.cpu()
            else:
                self.cross_attn_sum += video_to_text_attn.cpu()
            self.cross_attn_count += 1

            attn_map = self.cross_attn_sum / self.cross_attn_count
        else:
            if self.layer_cross_attn_sum[layer_idx] is None:
                self.layer_cross_attn_sum[layer_idx] = video_to_text_attn
            else:
                self.layer_cross_attn_sum[layer_idx] += video_to_text_attn
            self.layer_cross_attn_count[layer_idx] += 1

            attn_map = self.layer_cross_attn_sum[layer_idx] / self.layer_cross_attn_count[layer_idx]

        B, HWT, T = attn_map.shape
        F = HWT // (GlobalVars.HEIGHT * GlobalVars.WIDTH)
        attn_map = attn_map.reshape(B, F, GlobalVars.HEIGHT, GlobalVars.WIDTH, T)

        if isinstance(token_idx, (list)):
            attn_map = attn_map[..., token_idx]
            attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
        else:
            attn_map = attn_map[..., token_idx:token_idx+1].squeeze(-1)

        # Get min and max values using PyTorch
        attn_min = attn_map.amin(dim=(2, 3), keepdim=True)
        attn_max = attn_map.amax(dim=(2, 3), keepdim=True)
        
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
        
        return normalized_attn

    def get_flux_value_to_text_mask_by_mean(self, attn_map_mean, layer_idx=None, token_idx=[1]):
        text_length = 512
        text_to_video_attn = attn_map_mean[:, text_length:, :text_length]

        if text_to_video_attn.shape[1] != GlobalVars.HEIGHT * GlobalVars.WIDTH:
            text_to_video_attn = text_to_video_attn[:, :GlobalVars.HEIGHT * GlobalVars.WIDTH]

        if layer_idx is None:
            if self.cross_attn_sum is None:
                self.cross_attn_sum = text_to_video_attn
            else:
                self.cross_attn_sum += text_to_video_attn
            self.cross_attn_count += 1

            attn_map = self.cross_attn_sum / self.cross_attn_count
        else:
            if self.layer_cross_attn_sum[layer_idx] is None:
                self.layer_cross_attn_sum[layer_idx] = text_to_video_attn
            else:
                self.layer_cross_attn_sum[layer_idx] += text_to_video_attn
            self.layer_cross_attn_count[layer_idx] += 1

            attn_map = self.layer_cross_attn_sum[layer_idx] / self.layer_cross_attn_count[layer_idx]

        B, _, T = attn_map.shape
        attn_map = attn_map.reshape(B, GlobalVars.HEIGHT, GlobalVars.WIDTH, T)

        if isinstance(token_idx, (list)):
            attn_map = attn_map[..., token_idx]
            attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
        else:
            attn_map = attn_map[..., token_idx:token_idx+1].squeeze(-1)

        # Get min and max values using PyTorch
        attn_min = attn_map.amin(dim=(1, 2), keepdim=True)
        attn_max = attn_map.amax(dim=(1, 2), keepdim=True)
        
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
        
        return normalized_attn

    def get_text_to_value_mask_by_mean(self, attn_map_mean, layer_idx=None, token_idx=[1]):
        text_to_video_attn = attn_map_mean[:, -GlobalVars.TEXT_LENGTH:, :-GlobalVars.TEXT_LENGTH]

        if layer_idx is None:
            if self.cross_attn_sum is None:
                self.cross_attn_sum = text_to_video_attn
            else:
                self.cross_attn_sum += text_to_video_attn
            self.cross_attn_count += 1

            attn_map = self.cross_attn_sum / self.cross_attn_count
        else:
            if self.layer_cross_attn_sum[layer_idx] is None:
                self.layer_cross_attn_sum[layer_idx] = text_to_video_attn
            else:
                self.layer_cross_attn_sum[layer_idx] += text_to_video_attn
            self.layer_cross_attn_count[layer_idx] += 1

            attn_map = self.layer_cross_attn_sum[layer_idx] / self.layer_cross_attn_count[layer_idx]

        B, T, _ = attn_map.shape
        attn_map = attn_map.reshape(B, T, GlobalVars.HEIGHT, GlobalVars.WIDTH)

        if isinstance(token_idx, (list)):
            attn_map = attn_map[:, token_idx, :, :]
            attn_map = attn_map.sum(dim=1)  # Sum over selected tokens
        else:
            attn_map = attn_map[:, token_idx:token_idx+1, :, :].squeeze(1)

        # Get min and max values using PyTorch
        attn_min = attn_map.amin(dim=(1, 2), keepdim=True)
        attn_max = attn_map.amax(dim=(1, 2), keepdim=True)
        
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
        
        return normalized_attn

    def get_value_to_text_mask(self, q, k, layer_idx=None, token_idx=[1]):
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize result tensor (on GPU)
        attn_map_mean = torch.zeros(batch_size, seq_len, seq_len, device=q.device, dtype=q.dtype)
        
        # Parameters for batch computation
        batch_chunk = 1  # Number of batches to process at a time
        head_chunk = 1   # Number of attention heads to process at a time, can be adjusted based on GPU memory
        for i in range(0, batch_size, batch_chunk):
            for j in range(0, num_heads, head_chunk):
                # Select data for current batch and attention heads
                q_chunk = q[i:i+batch_chunk, j:j+head_chunk]
                k_chunk = k[i:i+batch_chunk, j:j+head_chunk]
                
                # Calculate attention scores for current attention heads
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                
                # Add to mean attention map
                attn_map_mean[i:i+batch_chunk] += attn_probs.sum(dim=1)
        
        # Calculate mean
        attn_map_mean /= num_heads
        video_to_text_attn = attn_map_mean[:, :-GlobalVars.TEXT_LENGTH, -GlobalVars.TEXT_LENGTH:]

        if layer_idx is None:
            if self.cross_attn_sum is None:
                self.cross_attn_sum = video_to_text_attn
            else:
                self.cross_attn_sum += video_to_text_attn
            self.cross_attn_count += 1

            attn_map = self.cross_attn_sum / self.cross_attn_count
        else:
            if self.layer_cross_attn_sum[layer_idx] is None:
                self.layer_cross_attn_sum[layer_idx] = video_to_text_attn
            else:
                self.layer_cross_attn_sum[layer_idx] += video_to_text_attn
            self.layer_cross_attn_count[layer_idx] += 1

            attn_map = self.layer_cross_attn_sum[layer_idx] / self.layer_cross_attn_count[layer_idx]

        B, _, T = attn_map.shape
        attn_map = attn_map.reshape(B, GlobalVars.HEIGHT, GlobalVars.WIDTH, T)

        if isinstance(token_idx, (list)):
            attn_map = attn_map[..., token_idx]
            attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
        else:
            attn_map = attn_map[..., token_idx:token_idx+1].squeeze(-1)

        # Get min and max values using PyTorch
        attn_min = attn_map.amin(dim=(1, 2), keepdim=True)
        attn_max = attn_map.amax(dim=(1, 2), keepdim=True)
        
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
        
        return normalized_attn