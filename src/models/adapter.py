import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.pooling import Attention1dPoolingHead, MeanPoolingHead, LightAttentionPoolingHead
from src.models.pooling import MeanPooling, MeanPoolingProjection

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], : x.shape[-1]]
    sin = sin[:, :, : x.shape[-2], : x.shape[-1]]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        assert (
            self.attention_head_size * config.num_attention_heads == config.hidden_size
        ), "Embed size needs to be divisible by num heads"
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, attention_mask=None, output_attentions=False):
        key_layer = self.transpose_for_scores(self.key_proj(key))
        value_layer = self.transpose_for_scores(self.value_proj(value))
        query_layer = self.transpose_for_scores(self.query_proj(query))
        query_layer = query_layer * self.attention_head_size**-0.5
        
        query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else context_layer
        
        return outputs


class ConservationCNN(nn.Module):
    def __init__(self):
        super(ConservationCNN, self).__init__()
        # 定义卷积核尺寸和膨胀系数
        kernel_sizes = [3, 5, 7]
        dilations = [1, 2, 4]
        num_filters = 64  # 每个卷积层的过滤器数量

        self.conv_layers = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            padding = ((k - 1) // 2) * d
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=k, dilation=d, padding=padding),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU()
                )
            )

    def forward(self, logits, embedding):
        # logits: (batch_size, 1, L)
        conv_outputs = []
        for conv in self.conv_layers:
            x = conv(logits)  # (batch_size, num_filters, L)
            conv_outputs.append(x)
        
        # 拼接卷积特征
        conv_features = torch.cat(conv_outputs, dim=1).transpose(1, 2)  # (batch_size, L, num_filters * len(kernel_sizes))

        # 融合卷积特征与嵌入
        combined_features = torch.cat([conv_features, embedding], dim=-1)  # (batch_size, L, total_features)

        return combined_features

class AdapterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if 'foldseek_seq' in config.structure_seqs:
            self.foldseek_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.cross_attention_foldseek = CrossModalAttention(config)
            
        if 'esm3_structure_seq' in config.structure_seqs:
            self.esm3_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.cross_attention_esm3 = CrossModalAttention(config)

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.layer_norm_with_ez = nn.LayerNorm(config.hidden_size + 8)
        # self.layer_norm_ez = nn.LayerNorm(8)
        # self.entropy_conv = ConservationCNN()
        
        if config.pooling_method == 'attention1d':
            self.classifier = Attention1dPoolingHead(config.hidden_size + 8, config.num_labels, config.pooling_dropout, config.return_attentions)
        elif config.pooling_method == 'mean':
            if "PPI" in config.dataset:
                self.pooling = MeanPooling()
                self.projection = MeanPoolingProjection(config.hidden_size + 8, config.num_labels, config.pooling_dropout)
            else:
                self.classifier = MeanPoolingHead(config.hidden_size + 8, config.num_labels, config.pooling_dropout)
        elif config.pooling_method == 'light_attention':
            self.classifier = LightAttentionPoolingHead(config.hidden_size + 8, config.num_labels, config.pooling_dropout)
        else:
            raise ValueError(f"classifier method {config.pooling_method} not supported")
    
    @torch.no_grad()
    def plm_embedding(self, plm_model, aa_input_ids, attention_mask):
        outputs = plm_model(input_ids=aa_input_ids, attention_mask=attention_mask)
        try:
            seq_embeds = outputs.last_hidden_state
            # logits = None
        except Exception as e:
            print(e)
            logits = outputs.logits
            # get entropy of logits
            logits = F.softmax(logits, dim=-1)
            logits = -torch.sum(logits * torch.log(logits), dim=-1).unsqueeze(1)
            logits = logits * attention_mask.unsqueeze(1)
            seq_embeds = outputs.hidden_states[-1]
            
        logits = None
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds, logits
    
    def forward(self, plm_model, batch):
        aa_input_ids, attention_mask = batch['aa_input_ids'], batch['attention_mask']
        seq_embeds, seq_logits = self.plm_embedding(plm_model, aa_input_ids, attention_mask)
        seq_logits = None

        if 'ez_descriptor' in self.config.structure_seqs:
            e_embeds, z_embeds = batch['e_descriptor_embeds'], batch['z_descriptor_embeds']
            ez_embeds = torch.cat([e_embeds, z_embeds], dim=-1)
            # ez_embeds = self.layer_norm_ez(ez_embeds)
            # concatenate ez descriptor embeddings with sequence embeddings
            # embeds = torch.cat([seq_embeds, ez_embeds], dim=-1)

        if 'aac' in self.config.structure_seqs:
            aac_embeds = batch['aac_embeds']
            embeds = torch.cat([seq_embeds, aac_embeds], dim=-1)

        if 'foldseek_seq' in self.config.structure_seqs:
            foldseek_seq = batch['foldseek_input_ids']
            foldseek_embeds = self.foldseek_embedding(foldseek_seq)
            # if 'ez_descriptor' in self.config.structure_seqs or 'aac' in self.config.structure_seqs:
            #     # cross attention with sequence and ez descriptor
            #     foldseek_embeds = self.cross_attention_foldseek(foldseek_embeds, embeds, embeds, attention_mask)
            #     embeds = foldseek_embeds + embeds
            # else:
                # cross attention with sequence
            foldseek_embeds = self.cross_attention_foldseek(foldseek_embeds, seq_embeds, seq_embeds, attention_mask)
            # foldseek_embeds = foldseek_embeds + seq_embeds
            # foldseek_embeds = self.layer_norm(foldseek_embeds)
        
        if 'esm3_structure_seq' in self.config.structure_seqs:
            esm3_seq = batch['esm3_structure_input_ids']
            esm3_embeds = self.esm3_embedding(esm3_seq)
            # if 'ez_descriptor' in self.config.structure_seqs or 'aac' in self.config.structure_seqs or 'foldseek_seq' in self.config.structure_seqs:
            #     # cross attention with sequence and ez descriptor
            #     esm3_embeds = self.cross_attention_esm3(esm3_embeds, embeds, embeds, attention_mask)
            #     embeds = esm3_embeds + embeds
            # else:
                # cross attention with sequence
            esm3_embeds = self.cross_attention_esm3(esm3_embeds, seq_embeds, seq_embeds, attention_mask)
            # esm3_embeds = esm3_embeds + seq_embeds
            # esm3_embeds = self.layer_norm(esm3_embeds)
        
        embeds = foldseek_embeds + esm3_embeds + seq_embeds
        embeds = self.layer_norm(embeds)
        if 'ez_descriptor' in self.config.structure_seqs:
            embeds = torch.cat([embeds, ez_embeds], dim=-1)
            embeds = self.layer_norm_with_ez(embeds)
        
        if 'ez_descriptor' in self.config.structure_seqs or 'aac' in self.config.structure_seqs or 'foldseek_seq' in self.config.structure_seqs:
            if seq_logits is not None:
                embeds = self.entropy_conv(seq_logits, embeds)
            
            if self.config.return_attentions and self.config.pooling_method == 'attention1d':
                logits, attn_wights = self.classifier(embeds, attention_mask)
                return logits, attn_wights
            else:
                logits = self.classifier(embeds, attention_mask)
        else:
            if seq_logits is not None:
                logits = self.entropy_conv(seq_logits, seq_embeds)
                
            if self.config.return_attentions and self.config.pooling_method == 'attention1d':
                logits, attn_wights = self.classifier(seq_embeds, attention_mask)
                return logits, attn_wights
            else:
                logits = self.classifier(seq_embeds, attention_mask)
                
        return logits

