'''
Description:不使用图神经网络，使用 ot注意力 来更新
Author: chenxuanqi
Date: 2025-05-09 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from models.ssegcn_bert_2 import LayerNorm, clones


class SSEGCNBertClassifier_with_getAtten(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.attdim, opt.polarities_dim)

        # 投影头用于对比学习（如 SimCLR 风格）
        self.projection = nn.Sequential(
            nn.Linear(self.opt.attdim, self.opt.attdim),
            nn.ReLU(),
            nn.Linear(self.opt.attdim, int(self.opt.attdim / 2))  # 低维空间用于对比损失
        )

    def forward(self, inputs, contrastive=False):
        """
        inputs: 原始输入
        contrastive: 如果为 True，返回投影用于对比学习
        """
        outputs1 = self.gcn_model(inputs)  # [B, 100]
        logits = self.classifier(outputs1)  # 分类输出

        if contrastive:
            contrast_vec = F.normalize(self.projection(outputs1), dim=-1)  # [B, 50]
            return logits, contrast_vec  # 返回对比向量用于 contrastive loss
        else:
            return logits, None


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert_with_OT(bert, opt, opt.num_layers)

    def forward(self, inputs):
        """
            text_bert_indices：文本的 BERT 输入索引，[batch_size, max_seq_len]，表示句子的单词或子词索引
            bert_segments_ids：BERT 的段 ID，于区分句子对（如 0 表示第一句，1 表示第二句），形状为 [batch_size, max_seq_len]
            attention_mask：BERT 的注意力掩码，标记有效 token，形状为 [batch_size, max_seq_len]
        """
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, src_mask, aspect_mask, short_mask = inputs
        h = self.gcn(inputs)  # [B, L, D]
        # 计算每个样本的方面词数量
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]
        # 扩展 aspect_mask 的维度，[batch_size, 1, D]
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.attdim)
        # 执行池化操作，提取方面相关的特征,[batch_size, hidden_dim]
        outputs1 = (h * aspect_mask).sum(dim=1) / asp_wn
        return outputs1


class GCNBert_with_OT(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert_with_OT, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.attdim = opt.attdim  # 固定的注意力维度
        # 定义 GCN 层的输出维度,为 bert 的一半
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.attn_drop = nn.Dropout(opt.attn_dropout)
        self.bert_layernorm = LayerNorm(opt.bert_dim)
        self.attn_layernorm = LayerNorm(self.attdim)

        self.W = nn.Linear(self.attdim, self.attdim)
        self.Wx = nn.Linear(self.attention_heads + self.attdim * 2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim, 50)
        self.aggregate_W = nn.Linear(self.attdim * 2, self.attdim)

        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim, short_dropout=opt.short_drop, alpha=self.opt.alpha
                                       , ot_epsilon=self.opt.ot_epsilon)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            # 第一层输入的维度为 bert_dim,其他层为输入输出均为 mem_dim
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        # 可学习的矩阵
        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, inputs):
        """
            text_bert_indices：BERT 的 token 索引，形状为 [batch_size, max_seq_len]。
            bert_segments_ids：段 ID，区分句子对，形状为 [batch_size, max_seq_len]。
            attention_mask：BERT 注意力掩码，标记有效 token，形状为 [batch_size, max_seq_len]。
            asp_start：方面起始位置索引，形状为 [batch_size] 或 [batch_size, 1]。
            asp_end：方面结束位置索引，形状同 asp_start。
            src_mask：源序列掩码，形状为 [batch_size, max_seq_len]，用于 GCN 或注意力。
            aspect_mask：方面掩码，标记方面相关 token，形状为 [batch_size, max_seq_len]。
            short_mask：短序列掩码，形状可能为 [batch_size, heads, max_seq_len, max_seq_len]，用于限制注意力范围。
        """
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, src_mask, aspect_mask, short_mask = inputs
        # [batch_size, max_seq_len] 扩展为 [batch_size, 1, max_seq_len]
        src_mask = src_mask.unsqueeze(-2)
        batch = src_mask.size(0)
        len = src_mask.size()[2]

        # 得到 bert 的输出
        outputs = self.bert(input_ids=text_bert_indices,
                            attention_mask=attention_mask,
                            token_type_ids=bert_segments_ids)
        sequence_output = outputs.last_hidden_state  # [B, L, D]
        pooled_output = outputs.pooler_output
        # 归一化
        sequence_output = self.bert_layernorm(sequence_output)  # [B, L, D]

        sequence_output = self.bert_drop(sequence_output)  # [B, L, D]
        pooled_output = self.pooled_drop(pooled_output)

        # [batch_size, max_seq_len, attdim]
        sequence_output = self.Wxx(sequence_output)

        # 计算每个样本的方面词数量 [batch_size, 1]
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        #  [batch_size, max_seq_len, attdim]
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.attdim)
        # 屏蔽非方面的 token,并池化 [batch_size, attdim]，这样每个维度只保留 aspect 的值
        aspect = (sequence_output * aspect_mask).sum(dim=1) / asp_wn

        for i in range(self.layers):
            # 调用多头注意力模块 attn，生成注意力权重，形状为 [B, head, seq_len, h_d]
            attn_output = self.attn(sequence_output, sequence_output, short_mask, aspect, src_mask)

            # 1. 交换维度 H 和 L: [B, L, H, D_k]
            attn_output = attn_output.permute(0, 2, 1, 3)

            # 2. 合并 H 和 D_k 维度: [B, L, H * D_k]
            attn_output = attn_output.contiguous().view(batch, len, self.attdim)

            sequence_output = self.attn_layernorm(sequence_output + self.attn_drop(attn_output))

        sequence_output = F.relu(sequence_output)

        return sequence_output


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, short_dropout=0.1, alpha=0.8, ot_epsilon=1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        # 克隆两个线性层
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        # 定义一个可学习的权重矩阵 weight_m，形状为 [h, d_k, d_k]
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        # 可学习的偏置项
        self.bias_m = nn.Parameter(torch.Tensor(1))
        # 定义一个线性层，将方面特征的维度从 d_model 映射到 d_k
        self.dense = nn.Linear(d_model, self.d_k)
        # 将 sentence_vec 和 aspect_vec 的隐藏层维度映射到 1
        self.mu_nu = nn.Linear(self.d_k, 1)
        self.alpha = alpha
        self.short_out = nn.Dropout(short_dropout)
        self.ot_epsilon = ot_epsilon

    def compute_ot_attention(self, sentence_vec, aspect_vec, mask, max_iter=50, tol=1e-6):
        """
        使用 Sinkhorn 最优传输，从句子向量到方面词向量计算注意力 π。

        参数：
            sentence_vec: [B, H, L, D] - 输入 token 表示
            aspect_vec:   [B, H, 1, D] - 单个方面词表示
            mask：[B, 1, 1, L]
            blur: float - Sinkhorn 模糊度（正则项）

        返回：
            pi: [B, H, L, 1] - 最优传输注意力分布
        """

        # 1. 计算余弦相似度距离矩阵 C: [B, H, L, 1]
        # 扩展维度以进行广播
        src = sentence_vec  # [B, H, L, D]
        tgt = aspect_vec  # [B, H, 1, D]
        # cost = torch.norm(src - tgt, dim=-1, keepdim=True)  # [B, H, L, 1]

        # 归一化后计算相似度 [B, H, L, 1]
        src_norm = F.normalize(src, dim=-1)
        tgt_norm = F.normalize(tgt, dim=-1)
        cosine_sim = torch.sum(src_norm * tgt_norm, dim=-1, keepdim=True)  # [B, H, L, 1]
        # 转换为 cost：相似度越大，代价越小
        cost = 1 - cosine_sim  # 或者 cost = -cosine_sim

        # 2. 初始化 mu 和 nu（概率分布）
        mu_logits = self.mu_nu(sentence_vec)  # [B, H, L, 1]
        # 处理 mask 以匹配维度
        mask = mask.permute(0, 1, 3, 2)  # [B, 1, L, 1]
        mask = mask.expand(-1, sentence_vec.size(1), -1, -1)  # [B, H, L, 1]
        mu_logits = mu_logits.masked_fill(mask == 0, -1e9)
        mu = F.softmax(mu_logits, dim=-2)  # [B, H, L, 1]

        nu_logits = self.mu_nu(aspect_vec)  # [B, H, 1, 1]
        nu = F.softmax(nu_logits, dim=-2)

        # 3. 初始化对偶变量 u, v
        u = torch.ones_like(mu)  # [B, H, L, 1]
        v = torch.ones_like(nu)  # [B, H, 1, 1]

        # 4. 计算 K = exp(-C / epsilon)
        K = torch.exp(-cost / self.ot_epsilon)  # [B, H, L, 1]

        # 5. Sinkhorn 迭代更新 u, v
        for _ in range(max_iter):
            u_prev = u
            # 更新 u, v
            u = mu / (K @ v + 1e-8)  # [B, H, L, 1]
            v = nu / (K.transpose(-2, -1) @ u + 1e-8)  # [B, H, 1, 1]
            if torch.max(torch.abs(u - u_prev)) < tol:
                break

        # 6. 恢复 transport plan π = u * K * v
        pi = u * K * v  # [B, H, L, 1]
        return pi

    def attention(self, query, key, short, aspect, bias_m, seq_h, asp_h, mask=None, alpha=0.8):
        """
            query: [batch_size, h, seq_len, d_k]
            key: [batch_size, h, seq_len, d_k]
            aspect：[batch_size, h, seq_len, d_k]  [batch_size, h, 1, d_k]
            short: [B, H, L, L]
            weight_m: [H, d_k, d_k]
            bias_m: [H, 1, 1]
            mask: [B, 1, 1, L]
        """
        B, H, L, D_K = query.size()
        L = query.size(-2)
        # 计算矩阵相乘 q * k，并缩放
        sentence_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D_K)  # [B, h, L, L]
        if mask is not None:
            # 将无效位置的分数设为 -1e9（接近负无穷），确保 softmax 后权重接近 0
            sentence_scores = sentence_scores.masked_fill(mask == 0, -1e9)
        sentence_scores = F.softmax(torch.add(sentence_scores, short), dim=-1)
        sentence_attn_output = torch.matmul(sentence_scores, seq_h)  # [B, H, L, D]

        # 使用 OT 计算分数
        pi = self.compute_ot_attention(seq_h, asp_h, mask)  # [B, H, L, 1]

        aspect_aware_rep = (pi * seq_h)  # [B, H, L, D]
        # aspect_aware_rep_expanded = aspect_aware_rep.unsqueeze(2).expand(-1, -1, L, -1)  # [B, H, L, D]

        output = alpha * sentence_attn_output + (1 - alpha) * aspect_aware_rep + bias_m

        return output

    def forward(self, query, key, short, aspect, mask=None):
        nbatches = query.size(0)
        # 对short进行 dropout
        short = self.short_out(short)

        seq_h = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        asp_h = aspect.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # [B, H, 1, d_k]

        # 保留 mask 的前 seq_len 个位置
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]

        # 对 query 和 key 进行线性变化后变形，[batch_size, h, seq_len, d_k]
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        # 映射到同一个空间，然后变换
        aspect = self.linears[0](aspect)  # [batch_size, D]
        aspect = aspect.view(nbatches, self.h, self.d_k).unsqueeze(2)
        attn_output = self.attention(query, key, short, aspect, self.bias_m, seq_h, asp_h, mask=mask, alpha=self.alpha)

        return attn_output  # [batch_size, h, seq_len, d_k]
