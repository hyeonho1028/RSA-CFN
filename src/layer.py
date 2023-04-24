import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        if hidden_size is None:
            self.hidden_size = input_size
        else:
            self.hidden_size = hidden_size
        self.Q = nn.Linear(self.input_size, self.hidden_size)
        self.K = nn.Linear(self.input_size, self.hidden_size)
        self.V = nn.Linear(self.input_size, self.input_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        scores = torch.matmul(q, k.transpose(1, 2))
        scores = scores / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v).squeeze(-1)

        return output, attn_weights