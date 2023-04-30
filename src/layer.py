import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################
################          Self-Attnetion         ################
#################################################################
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


######################################################################
###################            MLP-Mixer            ##################
# https://github.com/ISDS-Human-Understanding/HumanUnderstandingOpen #
######################################################################
class MlpBlock(nn.Module):
    def __init__(self,input_dim,dropout=0.3):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim,input_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(input_dim,input_dim)

    def forward(self, x):
        y = self.fc(x)
        y = self.gelu(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y


class MixerBlock(nn.Module):
    def __init__(self, input_dim, sequence_length, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.modal_mixing = MlpBlock(input_dim, dropout)
        self.sequence_mixing = MlpBlock(sequence_length, dropout)

    def transpose(self,x):
        return x.permute(0,2,1)

    def forward(self, x):
        y = self.ln(x)
        y = self.transpose(y)
        y = self.sequence_mixing(y)
        y = self.transpose(y)
        x = x + y
        y = self.ln(y)
        y = self.modal_mixing(y)
        y = y+x
        return y


class MultiModalMixer(nn.Module):
    def __init__(self):
        super().__init__()


        sequence_length = 555

        self.audio_projection = nn.Linear(1024, 256)
        self.text_projection = nn.Linear(768, 256)

        self.m_blocks = nn.ModuleList([
            MixerBlock(256, sequence_length, 0.1) for i in range(1)
        ])

        self.ln = nn.LayerNorm(256)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 7)
        )

    def forward(self, wav, text):
        audio_hidden_states = self.audio_projection(wav)
        text_hidden_states = self.text_projection(text)
        if audio_hidden_states.size()[1] < 299:
            # padding
            pad = torch.Tensor([[[0]*256]*(299-audio_hidden_states.size()[1])]*6).to('cuda')
            audio_hidden_states = torch.cat([audio_hidden_states, pad], dim=1)
            
        x = torch.cat([text_hidden_states, audio_hidden_states], dim=1)
        for block in self.m_blocks:
            x = block(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        
        return x
    
    
