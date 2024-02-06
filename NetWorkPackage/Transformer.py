# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt


class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()        
        
        self.use_encodeTransformer = config.use_encodeTransformer
        if config.use_encodeTransformer:
            # 引入transformer中encoder的层
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            # 归一化
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            # self-sttention，获取encoder结果
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        self.use_promt = False
        if config.use_promt:
            self.use_promt = True
            self.promt_embedding = PromtEmbeddings(config=config)
            self.layer = nn.Linear(3072, 2048)

        # embeddings的值
        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 前馈层
    def forward(self, img, src, mask, pos_embed, tgt, tgt_mask, promt=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        # conv_att = self.CBMA_block(src)
        # src = conv_att + conv_att

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        tgt = self.embeddings(tgt).permute(1, 0, 2)
        if self.use_promt:
            promt_tgt = self.promt_embedding(promt).permute(1, 0, 2)
            tgt = self.layer(torch.cat( (promt_tgt, tgt), dim=-1))

        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        # axial_memory = self.axial_net(img)

        features = src
        # 经过transformer模块编码特征
        if self.use_encodeTransformer:
            memory = self.encoder( src, src_key_padding_mask=mask, pos=pos_embed )
            features = memory
        
        #特征直接相加
        # memory = self.conv2( src.permute(1,2,0) ).permute(2,0,1)
        # features = torch.cat((memory,axial_memory), dim=-1)

        # 使用加权求和
        # memory = src * self.resnet_weight.t() + self.resnet_bias
        # axial_memory_muti = axial_memory * self.axial_weight.t() + self.axial_bias
        # features = memory+axial_memory_muti

        hs = self.decoder(tgt, features, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask,
                          pos=pos_embed, query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
        
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # self.spec1 = SpectralGatingNetwork(2048,24,8)
        # self.spec2 = SpectralGatingNetwork(2048,24,8)
        # self.spec3 = SpectralGatingNetwork(2048,24,8)
        # self.spec4 = SpectralGatingNetwork(2048,24,8)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                seg_mask: Optional[Tensor] = None):
        output = src

        for idx, layer in enumerate(self.layers):
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, seg_mask=seg_mask)
            
            # if idx == 0:
            #     spec1_output = self.spec1(output,(24,8))
            #     output_1 = output + spec1_output
            #     spec2_output = self.spec2(output_1,(24,8))
            #     output = output_1 + spec2_output
            # elif idx == 1:
            #     spec3_output = self.spec3(output,(24,8))
            #     output_3 = output + spec3_output
            #     spec4_output = self.spec4(output_3,(24,8))
            #     output = output_3 + spec4_output
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        
        # self.seg_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.norm4 = nn.LayerNorm(d_model)
        # self.dropout4 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead,dropout=dropout)     # 多头注意力机制
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # self-attention
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 计算self-attention的值
    # 将self-attention值传入 前馈神经层 add+norm
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     seg_mask: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)        
        # if seg_mask is not None:
        #     src2 = self.seg_attn(query = self.with_pos_embed(seg_mask, pos),
        #                         key = self.with_pos_embed(src, pos),
        #                         value=src)[0]
        #     src = src + self.dropout4(src)
        #     src = self.norm4(src)
        
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    seg_mask: Optional[Tensor] = None):
        src2 = self.norm1(src)
        # if seg_mask is not None:
        #     src3 = self.seg_attn(query = self.with_pos_embed(seg_mask, pos),
        #                         key = self.with_pos_embed(src2, pos),
        #                         value=src2)[0]
        #     src2 = src2 + self.dropout4(src3)
        #     src2 = self.norm4(src2)
        
        
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                seg_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, seg_mask)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, seg_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is not None:
            if tensor.shape[-1] != 2048:
                tensor[..., -2048:] = tensor[..., -2048:] + pos
            else:
                tensor = tensor + pos
        return tensor
    def with_resolve_embed(self, tensor, pos):
        return tensor if pos is None else tensor * pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    # padding_mask，输入序列的"对齐"，实质是一个张量，boolean类型，false为要处理的地方
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        # 相加、加入位置向量，解决序列问题
        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PromtEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.prompt_size, config.hidden_dim // 2, padding_idx=config.pad_token_id)

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim // 2, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)

        # 相加、加入位置向量，解决序列问题
        embeddings = input_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# subsequent_mask 为了使decoder不看见未来的数据
def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_transformer(config):
    return Transformer(
        config,
        d_model=config.hidden_dim,
        dropout=config.dropout,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        normalize_before=config.pre_norm,
        return_intermediate_dec=False,
        
    )
