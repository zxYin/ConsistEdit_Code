import os
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, List, Dict, Any

from diffusers.models.embeddings import apply_rotary_emb
import math
from consistEdit.global_var import GlobalVars

from PIL import Image

def regiter_attention_editor_diffusers_cog_mask(model, use_old_version=False):
    """
    Register an attention editor to Diffuser Pipeline for CogVideoX
    """
    def ca_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            text_seq_length = encoder_hidden_states.size(1)

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            batch_size, sequence_length, _ = hidden_states.shape

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)
                    # Apply RoPE if needed
            if image_rotary_emb is not None:

                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not self.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

            if GlobalVars.GENERATE_MASK:
                if GlobalVars.TOKEN_IDS is not None:
                    mask = GlobalVars.get_attn_controller().get_video_value_to_text_mask(query, key, token_idx=GlobalVars.TOKEN_IDS)
                else:
                    raise ValueError("TOKEN_IDS is not set")

                if mask.shape[0] == 4:
                    mask_tensor = mask[2]  # (H, W)
                elif mask.shape[0] == 2:
                    mask_tensor = mask[1]
                else:
                    mask_tensor = mask[0]
            
                mask_tensor[mask_tensor > 0.1] = 1
                mask_tensor[mask_tensor <= 0.1] = 0
                mask_tensor = 1 - mask_tensor

                if GlobalVars.SAVE_MASK_TRIGGER and GlobalVars.CUR_ATT_LAYER == GlobalVars.NUM_ATT_LAYERS -1:
                    for t, one_mask in enumerate(mask_tensor):
                        colored_frame = Image.fromarray((one_mask.cpu().numpy() * 255).astype(np.uint8))
                        if GlobalVars.MASK_OUTPUT_PATH is not None:
                            colored_frame.save(os.path.join(GlobalVars.MASK_OUTPUT_PATH, f"{t}.png"))
                        else:
                            colored_frame.save(f"cog_mask_output/{t}.png")
            else:
                masks = []
                for t in range(13):
                    if GlobalVars.MASK_OUTPUT_PATH is not None:
                        mask_image = Image.open(os.path.join(GlobalVars.MASK_OUTPUT_PATH, f"{t}.png")).resize((GlobalVars.WIDTH, GlobalVars.HEIGHT))
                    else:
                        raise ValueError("MASK_OUTPUT_PATH is not set")

                    mask_tensor = torch.from_numpy(np.array(mask_image)).unsqueeze(-1).flatten(0) / 255
                    masks.append(mask_tensor)
                mask_tensor = torch.stack(masks, dim=0).flatten(0)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1).unsqueeze(-1).to(dtype=value.dtype, device=value.device)

                text_mask = torch.zeros((1, 1, text_seq_length, 1)).to(dtype=value.dtype, device=value.device)
                value_mask_tensor = torch.cat([text_mask, mask_tensor], dim=2)
                if GlobalVars.CUR_STEP in range(0, GlobalVars.TEST_QK_STEP):
                    mask_tensor = torch.ones_like(mask_tensor)
                mask_tensor = torch.cat([text_mask, mask_tensor], dim=2)

                if query.shape[0] == 4:
                    qu_s, qu_t, qc_s, qc_t = query.chunk(4)
                    qu_t = qu_s * mask_tensor + qu_t * (1 - mask_tensor)
                    qc_t = qc_s * mask_tensor + qc_t * (1 - mask_tensor)
                    query = torch.cat([qu_s, qu_t, qc_s, qc_t], dim=0)

                    ku_s, ku_t, kc_s, kc_t = key.chunk(4)
                    ku_t = ku_s * mask_tensor + ku_t * (1 - mask_tensor)
                    kc_t = kc_s * mask_tensor + kc_t * (1 - mask_tensor)
                    key = torch.cat([ku_s, ku_t, kc_s, kc_t], dim=0)

                    vu_s, vu_t, vc_s, vc_t = value.chunk(4)
                    vu_t = vu_s * value_mask_tensor + vu_t * (1 - value_mask_tensor)
                    vc_t = vc_s * value_mask_tensor + vc_t * (1 - value_mask_tensor)
                    value = torch.cat([vu_s, vu_t, vc_s, vc_t], dim=0)

            if use_old_version:
                hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            else:
                batch_size, num_heads, _, head_dim = query.shape
                head_chunk = 16

                hidden_states = torch.zeros_like(query)
                if query.shape[0] == 4:
                    bs_list = [[0, 2], [1, 3]]
                else:
                    bs_list = [[0], [1]]

                for j in range(0, num_heads, head_chunk):
                    for bs_indexes in bs_list:
                        # Initialize result tensor (on GPU)
                        q_chunk = query[bs_indexes, j:j+head_chunk]
                        k_chunk = key[bs_indexes, j:j+head_chunk]

                        # Calculate attention scores for current attention heads
                        attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)

                        attn_probs = torch.softmax(attn_scores, dim=-1)
                        hidden_states[bs_indexes, j:j+head_chunk] = torch.matmul(attn_probs, value[bs_indexes, j:j+head_chunk])

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )

            GlobalVars.CUR_ATT_LAYER += 1
            if GlobalVars.CUR_ATT_LAYER == GlobalVars.NUM_ATT_LAYERS:
                GlobalVars.CUR_ATT_LAYER = 0

            return hidden_states, encoder_hidden_states
        return forward

    def register_editor(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count)
        return count

    cross_att_count = 0
    for net_name, net in model.transformer.named_children():
        if "transformer_blocks" in net_name:
            cross_att_count += register_editor(net, 0)
    GlobalVars.NUM_ATT_LAYERS = cross_att_count
    print("transformer_blocks", cross_att_count)

def regiter_attention_editor_diffusers_cog(model):
    """
    Register an attention editor to Diffuser Pipeline for CogVideoX
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            text_seq_length = encoder_hidden_states.size(1)

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            batch_size, sequence_length, _ = hidden_states.shape

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

                    # Apply RoPE if needed
            if image_rotary_emb is not None:

                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not self.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

            qu_s, qu_t, qc_s, qc_t = query.chunk(4)
            qu_t[:, :, text_seq_length:] = qu_s[:, :, text_seq_length:]
            qc_t[:, :, text_seq_length:] = qc_s[:, :, text_seq_length:]
            query = torch.cat([qu_s, qu_t, qc_s, qc_t], dim=0)
            # query = torch.cat([qu_s, qu_s, qc_s, qc_s], dim=0)

            ku_s, ku_t, kc_s, kc_t = key.chunk(4)
            ku_t[:, :, text_seq_length:] = ku_s[:, :, text_seq_length:]
            kc_t[:, :, text_seq_length:] = kc_s[:, :, text_seq_length:]
            key = torch.cat([ku_s, ku_t, kc_s, kc_t], dim=0)

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )

            GlobalVars.CUR_ATT_LAYER += 1
            if GlobalVars.CUR_ATT_LAYER == GlobalVars.NUM_ATT_LAYERS:
                GlobalVars.CUR_ATT_LAYER = 0

            return hidden_states, encoder_hidden_states
        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.transformer.named_children():
        if "transformer_blocks" in net_name:
            cross_att_count += register_editor(net, 0, "transformer_blocks")
    GlobalVars.NUM_ATT_LAYERS = cross_att_count
    print("transformer_blocks", cross_att_count)

def regiter_attention_editor_diffusers_flux(model):
    """
    Register an attention editor to Diffuser Pipeline for FLUX
    """
    def ca_forward(self, is_dual_attn):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
            if encoder_hidden_states is not None:
                # `context` projections.
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

                # attention
                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:

                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            if GlobalVars.CUR_STEP in range(0, GlobalVars.TEST_QK_STEP) and not is_dual_attn:
                text_length = GlobalVars.TEXT_LENGTH
                q_s, q_t = query.chunk(2)
                q_t[:, :, text_length:] = q_s[:, :, text_length:]
                query = torch.cat([q_s, q_t], dim=0)

                k_s, k_t = key.chunk(2)
                k_t[:, :, text_length:] = k_s[:, :, text_length:]
                key = torch.cat([k_s, k_t], dim=0)

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] :],
                )

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
        return forward

    def transformer_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
        ):
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()

            hidden_states = self.x_embedder(hidden_states)

            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

            temb = (
                self.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.time_text_embed(timestep, guidance, pooled_projections)
            )

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.pos_embed(ids)

            if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
                joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

            for index_block, block in enumerate(self.transformer_blocks):
                GlobalVars.CUR_ATT_LAYER = index_block
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # hidden_states 1, 2304, 3072
            # encoder_hidden_states 1, 512, 3072
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                GlobalVars.CUR_ATT_LAYER = index_block
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

            hidden_states = self.norm_out(hidden_states, temb)
            output = self.proj_out(hidden_states)
            GlobalVars.CUR_STEP += 1

            return (output,)
        return forward


    def register_editor(net, count, is_dual_attn):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, is_dual_attn)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, is_dual_attn)
        return count

    model.transformer.forward = transformer_forward(model.transformer)

    single_att_count = 0
    dual_att_count = 0
    for net_name, net in model.transformer.named_children():
        if "single_transformer_blocks" in net_name:
            # pass
            single_att_count += register_editor(net, 0, False)
        elif "transformer_blocks" in net_name:
            dual_att_count += register_editor(net, 0, True)

    GlobalVars.NUM_ATT_LAYERS = single_att_count
    GlobalVars.NUM_ATT_2_LAYERS = dual_att_count
    print("single_transformer_blocks", single_att_count)
    print("transformer_blocks", dual_att_count)

def regiter_attention_editor_diffusers_flux_mask(model, use_old_version=False):
    """
    Register an attention editor to Diffuser Pipeline for FLUX
    """
    def ca_forward(self, is_dual_attn):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
            if encoder_hidden_states is not None:
                # `context` projections.
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:

                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            batch_size, num_heads, seq_len, head_dim = query.shape
            head_chunk = 4
            hidden_states = torch.zeros_like(query)
            text_length = GlobalVars.TEXT_LENGTH

            if not is_dual_attn:
                if GlobalVars.GENERATE_MASK:
                    attn_map_mean = torch.zeros(batch_size, seq_len, seq_len, device=query.device, dtype=query.dtype)
                    if query.shape[0] == 2:
                        bs_list = [[0], [1]]
                    else:
                        bs_list = [[0]]

                    for j in range(0, num_heads, head_chunk):
                        for bs_indexes in bs_list:
                            # Initialize result tensor (on GPU)
                            q_chunk = query[bs_indexes, j:j+head_chunk]
                            k_chunk = key[bs_indexes, j:j+head_chunk]

                            # Calculate attention scores for current attention heads
                            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)

                            attn_probs = torch.softmax(attn_scores, dim=-1)
                            hidden_states[bs_indexes, j:j+head_chunk] = torch.matmul(attn_probs, value[bs_indexes, j:j+head_chunk])

                            attn_map_mean[bs_indexes] += attn_probs.sum(dim=1)

                    attn_map_mean /= num_heads
                    if GlobalVars.TOKEN_IDS is not None:
                        mask = GlobalVars.get_attn_controller().get_flux_value_to_text_mask_by_mean(attn_map_mean, token_idx=GlobalVars.TOKEN_IDS)
                    else:
                        raise ValueError("invalid token ids")

                    if GlobalVars.SAVE_MASK_TRIGGER and GlobalVars.CUR_ATT_LAYER == GlobalVars.NUM_ATT_LAYERS - 1:
                        mask_tensor = mask[0]  # (H, W)
                    
                        # Convert to PIL Image and apply magma colormap
                        mask_tensor[mask_tensor > 0.1] = 1
                        mask_tensor[mask_tensor <= 0.1] = 0
                        mask_tensor = 1 - mask_tensor
                        colored_frame = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8))
                        if GlobalVars.MASK_OUTPUT_PATH is not None:
                            colored_frame.save(GlobalVars.MASK_OUTPUT_PATH)
                        else:
                            raise ValueError("invalid mask output path")
                else:
                    if GlobalVars.MASK_OUTPUT_PATH is not None:
                        mask_image = Image.open(GlobalVars.MASK_OUTPUT_PATH).resize((GlobalVars.WIDTH, GlobalVars.HEIGHT))
                    else:
                        raise ValueError("invalid mask output path")
                    mask_tensor = torch.from_numpy(np.array(mask_image)).unsqueeze(-1).flatten(0) / 255
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1).unsqueeze(-1).to(dtype=value.dtype, device=value.device)

                    text_mask = torch.zeros((1, 1, text_length, 1)).to(dtype=value.dtype, device=value.device)
                    value_mask_tensor = torch.cat([text_mask, mask_tensor], dim=2)
                    if GlobalVars.CUR_STEP in range(0, GlobalVars.TEST_QK_STEP):
                        mask_tensor = torch.ones_like(mask_tensor)
                    mask_tensor = torch.cat([text_mask, mask_tensor], dim=2)

                    q_s, q_t = query.chunk(2)
                    q_t = q_s * mask_tensor + q_t * (1 - mask_tensor)
                    query = torch.cat([q_s, q_t], dim=0)

                    k_s, k_t = key.chunk(2)
                    k_t = k_s * mask_tensor + k_t * (1 - mask_tensor)
                    key = torch.cat([k_s, k_t], dim=0)

                    v_s, v_t = value.chunk(2)
                    v_t = v_s * value_mask_tensor + v_t * (1 - value_mask_tensor)
                    value = torch.cat([v_s, v_t], dim=0)

                    if use_old_version:
                        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
                    else:
                        for j in range(0, num_heads, head_chunk):
                            for bs_indexes in [[0], [1]]:
                                # Initialize result tensor (on GPU)
                                q_chunk = query[bs_indexes, j:j+head_chunk]
                                k_chunk = key[bs_indexes, j:j+head_chunk]

                                # Calculate attention scores for current attention heads
                                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)

                                attn_probs = torch.softmax(attn_scores, dim=-1)
                                hidden_states[bs_indexes, j:j+head_chunk] = torch.matmul(attn_probs, value[bs_indexes, j:j+head_chunk])
            else:
                hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] :],
                )

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
        return forward

    def transformer_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
        ):
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()

            hidden_states = self.x_embedder(hidden_states)

            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

            temb = (
                self.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.time_text_embed(timestep, guidance, pooled_projections)
            )

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.pos_embed(ids)

            if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
                joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

            for index_block, block in enumerate(self.transformer_blocks):
                GlobalVars.CUR_ATT_LAYER = index_block
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # hidden_states 1, 2304, 3072
            # encoder_hidden_states 1, 512, 3072
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                GlobalVars.CUR_ATT_LAYER = index_block
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

            hidden_states = self.norm_out(hidden_states, temb)
            output = self.proj_out(hidden_states)

            return (output,)
        return forward


    def register_editor(net, count, is_dual_attn):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, is_dual_attn)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, is_dual_attn)
        return count

    model.transformer.forward = transformer_forward(model.transformer)

    single_att_count = 0
    dual_att_count = 0
    for net_name, net in model.transformer.named_children():
        if "single_transformer_blocks" in net_name:
            # pass
            single_att_count += register_editor(net, 0, False)
        elif "transformer_blocks" in net_name:
            dual_att_count += register_editor(net, 0, True)

    GlobalVars.NUM_ATT_LAYERS = single_att_count
    GlobalVars.NUM_ATT_2_LAYERS = dual_att_count
    print("single_transformer_blocks", single_att_count)
    print("transformer_blocks", dual_att_count)

def regiter_attention_editor_diffusers_sd3(model):
    """
    Register an attention editor to Diffuser Pipeline for SD3
    """
    def ca_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            residual = hidden_states
            batch_size = hidden_states.shape[0]
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)
            
            # `context` projections.
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

                if GlobalVars.CUR_STEP in range(0, GlobalVars.TEST_QK_STEP):
                    if query.shape[0] == 2:
                        q_s, _ = query.chunk(2)
                        query = torch.cat([q_s, q_s], dim=0)

                        k_s, _ = key.chunk(2)
                        key = torch.cat([k_s, k_s], dim=0)

                    elif query.shape[0] == 4:
                        qu_s, _, qc_s, _ = query.chunk(4)
                        query = torch.cat([qu_s, qu_s, qc_s, qc_s], dim=0)

                        ku_s, _, kc_s, _ = key.chunk(4)
                        key = torch.cat([ku_s, ku_s, kc_s, kc_s], dim=0)
                    
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1] :],
                )
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
        return forward

    def transformer_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            block_controlnet_hidden_states: List = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
            skip_layers: Optional[List[int]] = None,
        ) :
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()

            height, width = hidden_states.shape[-2:]

            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
            temb = self.time_text_embed(timestep, pooled_projections)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

                joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

            for index_block, block in enumerate(self.transformer_blocks):
                # Skip specified layers
                is_skip = True if skip_layers is not None and index_block in skip_layers else False
                GlobalVars.CUR_ATT_LAYER = index_block

                if not is_skip:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )

            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )

            return (output, )
        return forward


    def register_editor(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count)
        return count

    cross_att_count = 0
    model.transformer.forward = transformer_forward(model.transformer)

    for net_name, net in model.transformer.named_children():
        if "transformer_blocks" in net_name:
            cross_att_count += register_editor(net, 0)
    GlobalVars.NUM_ATT_LAYERS = cross_att_count
    print("transformer_blocks", cross_att_count)

def regiter_attention_editor_diffusers_sd3_mask(model, use_old_version=False):
    """
    Register an attention editor with mask support to Diffuser Pipeline for SD3
    """
    def ca_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            residual = hidden_states
            batch_size = hidden_states.shape[0]
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # `context` projections.
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

                # Initialize result tensor (on GPU)
                if GlobalVars.GENERATE_MASK:
                    if GlobalVars.TOKEN_IDS is not None:
                        mask = GlobalVars.get_attn_controller().get_value_to_text_mask(query, key, token_idx=GlobalVars.TOKEN_IDS)
                    else:
                        raise ValueError("TOKEN_IDS is not set")

                    if mask.shape[0] == 4:
                        mask_tensor = mask[2]  # (H, W)
                    elif mask.shape[0] == 2:
                        mask_tensor = mask[1]
                    else:
                        mask_tensor = mask[0]
                
                    # Convert to PIL Image and apply magma colormap
                    mask_tensor[mask_tensor > 0.1] = 1
                    mask_tensor[mask_tensor <= 0.1] = 0
                    mask_tensor = 1 - mask_tensor

                    # # Convert to PIL Image and apply magma colormap
                    if GlobalVars.SAVE_MASK_TRIGGER and GlobalVars.CUR_ATT_LAYER == GlobalVars.NUM_ATT_LAYERS - 1:
                        colored_frame = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8))
                        if GlobalVars.MASK_OUTPUT_PATH is not None:
                            colored_frame.save(GlobalVars.MASK_OUTPUT_PATH)
                else:
                    if GlobalVars.MASK_OUTPUT_PATH is not None:
                        mask_image = Image.open(GlobalVars.MASK_OUTPUT_PATH).resize((GlobalVars.WIDTH, GlobalVars.HEIGHT))
                    else:
                        raise ValueError("MASK_OUTPUT_PATH is not set")

                    mask_tensor = torch.from_numpy(np.array(mask_image)).unsqueeze(-1).flatten(0) / 255
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1).unsqueeze(-1).to(dtype=value.dtype, device=value.device)
                    text_mask = torch.zeros((1, 1, encoder_hidden_states_query_proj.shape[2], 1)).to(dtype=value.dtype, device=value.device)

                    value_mask_tensor = torch.cat([mask_tensor, text_mask], dim=2)

                    if GlobalVars.CUR_STEP in range(0, GlobalVars.TEST_QK_STEP):
                        mask_tensor = torch.ones_like(mask_tensor)

                    mask_tensor = torch.cat([mask_tensor, text_mask], dim=2)

                    if query.shape[0] == 4:
                        qu_s, qu_t, qc_s, qc_t = query.chunk(4)
                        qu_t = qu_s * mask_tensor + qu_t * (1 - mask_tensor)
                        qc_t = qc_s * mask_tensor + qc_t * (1 - mask_tensor)
                        query = torch.cat([qu_s, qu_t, qc_s, qc_t], dim=0)

                        ku_s, ku_t, kc_s, kc_t = key.chunk(4)
                        ku_t = ku_s * mask_tensor + ku_t * (1 - mask_tensor)
                        kc_t = kc_s * mask_tensor + kc_t * (1 - mask_tensor)
                        key = torch.cat([ku_s, ku_t, kc_s, kc_t], dim=0)

                        vu_s, vu_t, vc_s, vc_t = value.chunk(4)
                        vu_t = vu_s * value_mask_tensor + vu_t * (1 - value_mask_tensor)
                        vc_t = vc_s * value_mask_tensor + vc_t * (1 - value_mask_tensor)
                        value = torch.cat([vu_s, vu_t, vc_s, vc_t], dim=0)

                    elif query.shape[0] == 2:
                        q_s, q_t = query.chunk(2)
                        q_t = q_s * mask_tensor + q_t * (1 - mask_tensor)
                        query = torch.cat([q_s, q_t], dim=0)

                        ku_s, ku_t = key.chunk(2)
                        ku_t = ku_s * mask_tensor + ku_t * (1 - mask_tensor)
                        key = torch.cat([ku_s, ku_t], dim=0)

                        vu_s, vu_t = value.chunk(4)
                        vu_t = vu_s * value_mask_tensor + vu_t * (1 - value_mask_tensor)
                        value = torch.cat([vu_s, vu_t], dim=0)

            if use_old_version:
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            else:
                batch_size, num_heads, _, head_dim = query.shape
                head_chunk = 16

                hidden_states = torch.zeros_like(query)
                if query.shape[0] == 4:
                    bs_list = [[0, 2], [1, 3]]
                else:
                    bs_list = [[0], [1]]

                for j in range(0, num_heads, head_chunk):
                    for bs_indexes in bs_list:
                        # Initialize result tensor (on GPU)
                        q_chunk = query[bs_indexes, j:j+head_chunk]
                        k_chunk = key[bs_indexes, j:j+head_chunk]

                        # Calculate attention scores for current attention heads
                        attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)

                        attn_probs = torch.softmax(attn_scores, dim=-1)
                        hidden_states[bs_indexes, j:j+head_chunk] = torch.matmul(attn_probs, value[bs_indexes, j:j+head_chunk])

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1] :],
                )
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
        return forward

    def transformer_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            block_controlnet_hidden_states: List = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
            skip_layers: Optional[List[int]] = None,
        ) :
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()

            height, width = hidden_states.shape[-2:]

            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
            temb = self.time_text_embed(timestep, pooled_projections)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

                joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

            for index_block, block in enumerate(self.transformer_blocks):
                # Skip specified layers
                is_skip = True if skip_layers is not None and index_block in skip_layers else False
                GlobalVars.CUR_ATT_LAYER = index_block

                if not is_skip:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )

            return (output, )
        return forward


    def register_editor(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count)
        return count

    cross_att_count = 0
    model.transformer.forward = transformer_forward(model.transformer)

    for net_name, net in model.transformer.named_children():
        if "transformer_blocks" in net_name:
            cross_att_count += register_editor(net, 0)

    GlobalVars.NUM_ATT_LAYERS = cross_att_count
    print("transformer_blocks", cross_att_count)

def regiter_attention_editor_diffusers_sd3_mask_real(model, use_old_version=False):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """

    def get_map_info_name(type, step, layer, order=0):
        return f"{type}_layer{layer}_step{step}_order{order}"

    def ca_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            residual = hidden_states
            batch_size = hidden_states.shape[0]
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # `context` projections.
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

                save_step_idx = GlobalVars.TOTAL_STEPS - GlobalVars.CUR_STEP - 1 if GlobalVars.IS_INVERSE else GlobalVars.CUR_STEP
                save_order_idx = 0

                if GlobalVars.IS_INVERSE:
                    if query.shape[0] == 1:
                        qc_s = query[0]
                        qu_s = None
                        kc_s = key[0]
                        ku_s = None
                        vc_s = value[0]
                        vu_s = None
                        q_text_c_s = encoder_hidden_states_query_proj[0]
                        q_text_u_s = None
                        k_text_c_s = encoder_hidden_states_key_proj[0]
                        k_text_u_s = None
                        v_text_c_s = encoder_hidden_states_value_proj[0]
                        v_text_u_s = None
                    elif query.shape[0] == 2:
                        qu_s, qc_s = query.chunk(2)
                        ku_s, kc_s = key.chunk(2)
                        vu_s, vc_s = value.chunk(2)
                        q_text_u_s, q_text_c_s = encoder_hidden_states_query_proj.chunk(2)
                        k_text_u_s, k_text_c_s = encoder_hidden_states_key_proj.chunk(2)
                        v_text_u_s, v_text_c_s = encoder_hidden_states_value_proj.chunk(2)
                    else:
                        raise ValueError(f"query shape {query.shape} is not supported")

                    if qu_s != None:
                        GlobalVars.MAP_SAVER[get_map_info_name("qu_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = qu_s.cpu()
                        GlobalVars.MAP_SAVER[get_map_info_name("ku_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = ku_s.cpu()
                        GlobalVars.MAP_SAVER[get_map_info_name("vu_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = vu_s.cpu()
                        GlobalVars.MAP_SAVER[get_map_info_name("q_text_u_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = q_text_u_s.cpu()
                        GlobalVars.MAP_SAVER[get_map_info_name("k_text_u_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = k_text_u_s.cpu()
                        GlobalVars.MAP_SAVER[get_map_info_name("v_text_u_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = v_text_u_s.cpu()
                    GlobalVars.MAP_SAVER[get_map_info_name("qc_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = qc_s.cpu()
                    GlobalVars.MAP_SAVER[get_map_info_name("kc_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = kc_s.cpu()
                    GlobalVars.MAP_SAVER[get_map_info_name("vc_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = vc_s.cpu()

                    GlobalVars.MAP_SAVER[get_map_info_name("q_text_c_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = q_text_c_s.cpu()
                    GlobalVars.MAP_SAVER[get_map_info_name("k_text_c_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = k_text_c_s.cpu()
                    GlobalVars.MAP_SAVER[get_map_info_name("v_text_c_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx)] = v_text_c_s.cpu()
                else:
                    qu_s = GlobalVars.MAP_SAVER.get(get_map_info_name("qu_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    qc_s = GlobalVars.MAP_SAVER.get(get_map_info_name("qc_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    ku_s = GlobalVars.MAP_SAVER.get(get_map_info_name("ku_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    kc_s = GlobalVars.MAP_SAVER.get(get_map_info_name("kc_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    vu_s = GlobalVars.MAP_SAVER.get(get_map_info_name("vu_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    vc_s = GlobalVars.MAP_SAVER.get(get_map_info_name("vc_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)

                    q_text_u_s = GlobalVars.MAP_SAVER.get(get_map_info_name("q_text_u_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    q_text_c_s = GlobalVars.MAP_SAVER.get(get_map_info_name("q_text_c_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    k_text_u_s = GlobalVars.MAP_SAVER.get(get_map_info_name("k_text_u_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    k_text_c_s = GlobalVars.MAP_SAVER.get(get_map_info_name("k_text_c_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    v_text_u_s = GlobalVars.MAP_SAVER.get(get_map_info_name("v_text_u_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)
                    v_text_c_s = GlobalVars.MAP_SAVER.get(get_map_info_name("v_text_c_s", save_step_idx, GlobalVars.CUR_ATT_LAYER, save_order_idx), None)

                    if query.shape[0] == 1:
                        query = qc_s.to(dtype=query.dtype, device=query.device)
                        key = kc_s.to(dtype=key.dtype, device=key.device)
                        value = vc_s.to(dtype=value.dtype, device=value.device)
                        encoder_hidden_states_key_proj = k_text_c_s.to(dtype=encoder_hidden_states_key_proj.dtype, device=encoder_hidden_states_key_proj.device)
                        encoder_hidden_states_query_proj = q_text_c_s.to(dtype=encoder_hidden_states_query_proj.dtype, device=encoder_hidden_states_query_proj.device)
                        encoder_hidden_states_value_proj = v_text_c_s.to(dtype=encoder_hidden_states_value_proj.dtype, device=encoder_hidden_states_value_proj.device)
                    elif query.shape[0] == 2:
                        query[0] = qc_s.to(dtype=query.dtype, device=query.device)
                        key[0] = kc_s.to(dtype=key.dtype, device=key.device)
                    elif query.shape[0] == 4:
                        if qu_s is not None:
                            query[0] = qu_s.to(dtype=query.dtype, device=query.device)
                        query[2] = qc_s.to(dtype=query.dtype, device=query.device)

                        if ku_s is not None:
                            key[0] = ku_s.to(dtype=key.dtype, device=key.device)
                        key[2] = kc_s.to(dtype=key.dtype, device=key.device)

                        if vu_s is not None:
                            value[0] = vu_s.to(dtype=value.dtype, device=value.device)
                        value[2] = vc_s.to(dtype=value.dtype, device=value.device)
                    else:
                        raise ValueError(f"query shape {query.shape} is not supported")


                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

                if GlobalVars.IS_INVERSE:
                    GlobalVars.get_attn_controller().save_cur_attn_map(query, key, cur_step=GlobalVars.CUR_STEP, layer_id=GlobalVars.CUR_ATT_LAYER)
                    if GlobalVars.TOKEN_IDS is not None:
                        mask = GlobalVars.get_attn_controller().get_value_to_text_mask(query, key, token_idx=GlobalVars.TOKEN_IDS)
                    else:
                        raise ValueError("TOKEN_IDS is not set")

                    if GlobalVars.SAVE_MASK_TRIGGER and GlobalVars.CUR_ATT_LAYER == GlobalVars.NUM_ATT_LAYERS - 1:
                        mask_tensor = mask[0].float()  # (H, W)
                        
                        # # Convert to PIL Image and apply magma colormap
                        mask_tensor[mask_tensor > 0.1] = 1
                        mask_tensor[mask_tensor <= 0.1] = 0
                        mask_tensor = 1 - mask_tensor

                        colored_frame = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8))
                        if GlobalVars.MASK_OUTPUT_PATH is not None:
                            colored_frame.save(GlobalVars.MASK_OUTPUT_PATH)
                        else:
                            raise ValueError("MASK_OUTPUT_PATH is not set")
                else:
                    if GlobalVars.MASK_OUTPUT_PATH is None:
                        raise ValueError("MASK_OUTPUT_PATH is not set")
                    else:
                        mask_image = Image.open(GlobalVars.MASK_OUTPUT_PATH).resize((GlobalVars.WIDTH, GlobalVars.HEIGHT)).convert("RGB")
                    mask_tensor = torch.from_numpy(np.array(mask_image)[:, :, :1]).flatten(0) / 255

                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1).unsqueeze(-1).to(dtype=value.dtype, device=value.device)

                    mask_tensor[mask_tensor > 0.5] = 1
                    mask_tensor[mask_tensor <= 0.5] = 0

                    text_mask = torch.zeros((1, 1, encoder_hidden_states_query_proj.shape[2], 1)).to(dtype=value.dtype, device=value.device)
                    value_mask_tensor = torch.cat([mask_tensor, text_mask], dim=2)

                    if GlobalVars.CUR_STEP in range(0, GlobalVars.TEST_QK_STEP):
                        mask_tensor = torch.ones_like(mask_tensor)
                    mask_tensor = torch.cat([mask_tensor, text_mask], dim=2)

                    if query.shape[0] == 4:
                        qu_s, qu_t, qc_s, qc_t = query.chunk(4)
                        qu_t = qu_s * mask_tensor + qu_t * (1 - mask_tensor)
                        qc_t = qc_s * mask_tensor + qc_t * (1 - mask_tensor)
                        query = torch.cat([qu_s, qu_t, qc_s, qc_t], dim=0)

                        ku_s, ku_t, kc_s, kc_t = key.chunk(4)
                        ku_t = ku_s * mask_tensor + ku_t * (1 - mask_tensor)
                        kc_t = kc_s * mask_tensor + kc_t * (1 - mask_tensor)
                        key = torch.cat([ku_s, ku_t, kc_s, kc_t], dim=0)

                        vu_s, vu_t, vc_s, vc_t = value.chunk(4)
                        vu_t = vu_s * value_mask_tensor + vu_t * (1 - value_mask_tensor)
                        vc_t = vc_s * value_mask_tensor + vc_t * (1 - value_mask_tensor)
                        value = torch.cat([vu_s, vu_t, vc_s, vc_t], dim=0)
                    elif query.shape[0] == 2:
                        q_s, q_t = query.chunk(2)
                        q_t = q_s * mask_tensor + q_t * (1 - mask_tensor)
                        query = torch.cat([q_s, q_t], dim=0)

                        k_s, k_t = key.chunk(2)
                        k_t = k_s * mask_tensor + k_t * (1 - mask_tensor)
                        key = torch.cat([k_s, k_t], dim=0)
                        
                        v_s, v_t = value.chunk(2)
                        v_t = v_s * value_mask_tensor + v_t * (1 - value_mask_tensor)
                        value = torch.cat([v_s, v_t], dim=0)
                    else:
                        raise ValueError(f"query shape {query.shape} is not supported")

            if use_old_version:
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            else:
                batch_size, num_heads, _, head_dim = query.shape
                head_chunk = 16

                hidden_states = torch.zeros_like(query)
                if query.shape[0] == 4:
                    bs_list = [[0, 2], [1, 3]]
                elif query.shape[0] == 2:
                    bs_list = [[0], [1]]
                else:
                    bs_list = [[0]]

                for j in range(0, num_heads, head_chunk):
                    for bs_indexes in bs_list:
                        # Initialize result tensor (on GPU)
                        q_chunk = query[bs_indexes, j:j+head_chunk]
                        k_chunk = key[bs_indexes, j:j+head_chunk]

                        # Calculate attention scores for current attention heads
                        attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)

                        attn_probs = torch.softmax(attn_scores, dim=-1)
                        hidden_states[bs_indexes, j:j+head_chunk] = torch.matmul(attn_probs, value[bs_indexes, j:j+head_chunk])

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1] :],
                )
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
        return forward

    def transformer_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            block_controlnet_hidden_states: List = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
            skip_layers: Optional[List[int]] = None,
        ) :
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()

            height, width = hidden_states.shape[-2:]

            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
            temb = self.time_text_embed(timestep, pooled_projections)

            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

                joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

            for index_block, block in enumerate(self.transformer_blocks):
                # Skip specified layers
                is_skip = True if skip_layers is not None and index_block in skip_layers else False
                GlobalVars.CUR_ATT_LAYER = index_block

                if not is_skip:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )

            return (output, )
        return forward


    def register_editor(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count)
        return count

    cross_att_count = 0
    model.transformer.forward = transformer_forward(model.transformer)

    for net_name, net in model.transformer.named_children():
        if "transformer_blocks" in net_name:
            cross_att_count += register_editor(net, 0)
    GlobalVars.NUM_ATT_LAYERS = cross_att_count
    print("transformer_blocks", cross_att_count)