import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 2. 物理感知注意力 (考虑物理约束)
class PhysicsAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads, physics_dim=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.physics_dim = physics_dim
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 物理约束权重
        self.physics_weight = nn.Parameter(torch.randn(physics_dim, d_model))
        self.physics_bias = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, physics_info=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 标准注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 物理约束调整
        if physics_info is not None:
            # physics_info: [batch_size, seq_len, physics_dim]
            physics_constraint = torch.matmul(physics_info, self.physics_weight)  # [B, seq_len, d_model]
            physics_constraint = physics_constraint.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            physics_constraint = physics_constraint.view(batch_size, self.num_heads, -1, self.d_k)
            
            # 调整注意力分数
            physics_scores = torch.matmul(Q, physics_constraint.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores + self.physics_bias * physics_scores
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

# 6. 多尺度注意力 (适合多尺度物理现象)
class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, num_heads, scales=[1, 2, 4], dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        self.num_scales = len(scales)
        
        # 每个尺度的注意力
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in scales
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Linear(d_model * self.num_scales, d_model)
        
    def forward(self, query, key, value):
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale > 1:
                # 下采样
                seq_len = query.size(1)
                indices = torch.arange(0, seq_len, scale, device=query.device)
                scaled_query = query[:, indices, :]
                scaled_key = key[:, indices, :]
                scaled_value = value[:, indices, :]
            else:
                scaled_query = query
                scaled_key = key
                scaled_value = value
            
            # 注意力计算
            attn_out, _ = self.scale_attentions[i](scaled_query, scaled_key, scaled_value)
            
            if scale > 1:
                # 上采样回原始尺寸
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2), 
                    size=query.size(1), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(attn_out)
        
        # 融合多尺度特征
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(combined)
        
        return output





# 4. Multi-Physics Coupling Attention (多物理场耦合注意力)
class MultiPhysicsCouplingAttention(nn.Module):
    """处理多物理场耦合的attention机制"""
    
    def __init__(self, d_model, num_heads, physics_fields=['temperature', 'velocity', 'pressure'], dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.physics_fields = physics_fields
        self.num_fields = len(physics_fields)
        
        # 每个物理场的专用attention
        self.field_attentions = nn.ModuleDict({
            field: nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for field in physics_fields
        })
        
        # 场间耦合矩阵
        self.coupling_matrix = nn.Parameter(torch.eye(self.num_fields) + 0.1 * torch.randn(self.num_fields, self.num_fields))
        
        # 场识别网络
        self.field_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.num_fields),
            nn.Softmax(dim=-1)
        )
        
        # 输出融合
        self.field_fusion = nn.Linear(d_model * self.num_fields, d_model)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 识别每个位置的主导物理场
        field_probs = self.field_classifier(query)  # [B, seq_len, num_fields]
        
        # 为每个物理场计算attention
        field_outputs = {}
        for i, field in enumerate(self.physics_fields):
            field_attn_out, _ = self.field_attentions[field](query, key, value)
            field_outputs[field] = field_attn_out
        
        # 考虑场间耦合
        coupled_outputs = []
        for i, field in enumerate(self.physics_fields):
            coupled_output = torch.zeros_like(field_outputs[field])
            for j, other_field in enumerate(self.physics_fields):
                coupling_strength = self.coupling_matrix[i, j]
                coupled_output += coupling_strength * field_outputs[other_field]
            coupled_outputs.append(coupled_output)
        
        # 根据场概率加权融合
        final_output = torch.zeros_like(query)
        for i, field in enumerate(self.physics_fields):
            field_weight = field_probs[:, :, i:i+1]  # [B, seq_len, 1]
            final_output += field_weight * coupled_outputs[i]
        
        return final_output

