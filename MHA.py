# Associative Transformer Is A Sparse Representation Learner
# See https://arxiv.org/abs/2309.12862
#
# Author: Yuwei Sun
#
# The code is based on the 
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py



import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def cv_squared(self, x):
        
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, q, k, v, bottleneck, coe = 1e-2):

        loss = 0
        output = []
        gates = []
        #for each head
        for i in range(q.size(1)):
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))[0][i]
            
            # calculate topk + 1 that will be needed for the noisy gates
            top_logits, top_indices = attn.topk(min(bottleneck + 1, attn.size(1)), dim=-1)
            top_k_logits = top_logits[:, :bottleneck]
            top_k_indices = top_indices[:, :bottleneck]
            top_k_gates = F.softmax(top_k_logits, dim = -1)

            zeros = torch.zeros_like(attn, requires_grad=True)
            gate = zeros.scatter(1, top_k_indices, top_k_gates)
            output.append(torch.matmul(gate, v[0,i,:,:]))
            gates.append(gate)

            # calculate importance loss and load loss for balanced expert usage
            importance = gate.sum(0)
            loads = (gate > 0).sum(0)
            loss += (self.cv_squared(importance) + self.cv_squared(loads)) * coe
        
        #stack results of all heads
        output = torch.stack(output, dim = 1)
        gates = torch.stack(gates, dim = 1)

        return output, gates, loss
    

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_k, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_v, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_v, eps=1e-6)


    def forward(self, q, k, v, bottleneck, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(0).expand(n_head,*(mask.size(0),mask.size(1)))   # For head axis broadcasting.

        q, gates, loss = self.attention(q, k, v, bottleneck)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
  
        q = self.layer_norm(q)
        

        return q, gates, loss
    


class HopfieldReplaceMHA(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, _ = self.attention(q, k, v, v.size(1))

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
  
        return q, attn
