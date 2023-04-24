import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformers import XLMRobertaModel
from transformers import Wav2Vec2PreTrainedModel

from transformers import Wav2Vec2Model

from .layer import SelfAttention

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        return logits


class xlm_RoBertaBase(XLMRobertaModel):
    def __init__(self, config):
        super(xlm_RoBertaBase, self).__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
        embeddings = torch.mean(outputs, axis=1)
        embeddings = self.norm(embeddings)
        logits = self.classifier(embeddings)
        return logits


class MultiModalClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multimodal_method = config.multimodal_method
        self.num_classes = config.rnn_config.num_classes

        if self.multimodal_method=='base':
            concat_size = config.wav_config.hidden_size + config.rnn_config.hidden_size
            self.norm = nn.LayerNorm(concat_size)
            self.classifier = nn.Linear(concat_size, self.num_classes)
            
        elif self.multimodal_method=='late_fusion':
            self.wave_classifier = nn.Linear(config.wav_config.hidden_size, self.num_classes)
            self.text_classifier = nn.Linear(config.rnn_config.hidden_size, self.num_classes)
            self.norm = nn.LayerNorm(self.num_classes)

        elif self.multimodal_method=='stacking':
            concat_size = config.wav_config.hidden_size + config.rnn_config.hidden_size
            self.norm = nn.LayerNorm(concat_size)
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
            self.classifier = nn.Linear(concat_size, self.num_classes)

        elif self.multimodal_method=='residuals':
            concat_size = config.wav_config.hidden_size + config.rnn_config.hidden_size
            self.norm = nn.LayerNorm(concat_size)
            self.res_block = nn.Sequential(
                nn.Linear(concat_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.res_block2 = nn.Sequential(
                nn.Linear(concat_size+512, concat_size),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(concat_size, self.num_classes)
            
        elif self.multimodal_method=='residuals_attn':
            concat_size = config.wav_config.hidden_size + config.rnn_config.hidden_size
            self.attn = SelfAttention(1, 256)
            self.norm = nn.LayerNorm(concat_size)
            self.res_block = nn.Sequential(
                nn.Linear(concat_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.res_block2 = nn.Sequential(
                nn.Linear(concat_size+512, concat_size),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(concat_size, self.num_classes)
            
        elif self.multimodal_method=='rsa_cfn':
            self.wav_proj = nn.Linear(config.wav_config.hidden_size, 128)
            self.rnn_proj = nn.Linear(config.rnn_config.hidden_size, 128)
            
            self.wave_attn = SelfAttention(1, 96)
            self.text_attn = SelfAttention(1, 96)
            
            self.wav_norm = nn.LayerNorm(128)
            self.rnn_norm = nn.LayerNorm(128)
            
            cross_fusion_size = (128+1) * (128+1)
            
            self.res_block = nn.Sequential(
                nn.Linear(cross_fusion_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.res_block2 = nn.Sequential(
                nn.Linear(cross_fusion_size+1024, cross_fusion_size),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Sequential(
                nn.Linear(cross_fusion_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, self.num_classes)
            )
        

        elif self.multimodal_method=='hybrid_fusion':
            concat_size = config.wav_config.hidden_size + config.rnn_config.hidden_size
            
            # base
            self.base_norm = nn.LayerNorm(concat_size)
            self.base_classifier = nn.Linear(concat_size, self.num_classes)
            
            # late fusion
            self.wave_classifier = nn.Linear(config.wav_config.hidden_size, self.num_classes)
            self.text_classifier = nn.Linear(config.rnn_config.hidden_size, self.num_classes)
            self.lf_norm = nn.LayerNorm(self.num_classes)
            
            # residuals
            self.res_norm = nn.LayerNorm(concat_size)
            self.res_block = nn.Sequential(
                nn.Linear(concat_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.res_block2 = nn.Sequential(
                nn.Linear(concat_size+512, concat_size),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.res_classifier = nn.Linear(concat_size, self.num_classes)
            
            # residuals_attn_cf
            self.wav_proj = nn.Linear(config.wav_config.hidden_size, 128)
            self.rnn_proj = nn.Linear(config.rnn_config.hidden_size, 128)
            
            self.wave_attn = SelfAttention(1, 96)
            self.text_attn = SelfAttention(1, 96)
            
            self.wav_norm = nn.LayerNorm(128)
            self.rnn_norm = nn.LayerNorm(128)
            
            cross_fusion_size = (128+1) * (128+1)
            
            self.rac_res_block = nn.Sequential(
                nn.Linear(cross_fusion_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.rac_res_block2 = nn.Sequential(
                nn.Linear(cross_fusion_size+1024, cross_fusion_size),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.rac_classifier = nn.Sequential(
                nn.Linear(cross_fusion_size, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, self.num_classes)
            )
            
            self.hybrid_norm = nn.LayerNorm(self.num_classes)

    def forward(self, wav_vec, rnn_vec):
        if self.multimodal_method=='base':
            x = torch.cat([wav_vec, rnn_vec], dim=1)
            x = self.norm(x)
            outputs = self.classifier(x)
        
        elif self.multimodal_method=='late_fusion':
            wave_output = self.wave_classifier(wav_vec)
            text_output = self.text_classifier(rnn_vec)
            outputs = self.norm(torch.stack([wave_output, text_output], dim=1))
            outputs = torch.mean(outputs, dim=1)
            
        elif self.multimodal_method=='stacking':
            x = torch.cat([wav_vec, rnn_vec], dim=1)
            x = self.norm(x)
            for i, dropout in enumerate(self.dropouts):
                if i==0:
                    outputs = self.classifier(dropout(x))
                else:
                    outputs += self.classifier(dropout(x))
            else:
                outputs /= len(self.dropouts)

        elif self.multimodal_method=='residuals':
            x = torch.cat([wav_vec, rnn_vec], dim=1)
            x = self.norm(x)

            res = self.res_block(x)
            res_concat = torch.cat([x, res], dim=1)

            res2 = self.res_block2(res_concat)
            outputs = self.classifier(x + res2)

            return outputs
        
        elif self.multimodal_method=='residuals_attn':
            x = torch.cat([wav_vec, rnn_vec], dim=1)
            x = self.norm(x)

            attn_x, attn_weights = self.attn(x)
            
            res = self.res_block(attn_x)
            res_concat = torch.cat([attn_x, res], dim=1)

            res2 = self.res_block2(res_concat)
            outputs = self.classifier(x + res2)

            return outputs, attn_weights

        elif self.multimodal_method=='rsa_cfn':
            bs = wav_vec.size()[0]
            wav_vec_proj = self.wav_proj(wav_vec)
            rnn_vec_proj = self.rnn_proj(rnn_vec)
            
            wav_vec = self.wav_norm(self.wave_attn(wav_vec_proj)[0] + wav_vec_proj)
            rnn_vec = self.rnn_norm(self.text_attn(rnn_vec_proj)[0] + rnn_vec_proj)
            
            _wav = torch.cat((
                Variable(torch.ones(bs, 1).type(wav_vec.dtype).to(wav_vec.device), requires_grad=False), wav_vec), dim=1)
            _rnn = torch.cat((
                Variable(torch.ones(bs, 1).type(rnn_vec.dtype).to(rnn_vec.device), requires_grad=False), rnn_vec), dim=1)
            
            cross_fusion = torch.matmul(_wav.unsqueeze(2), _rnn.unsqueeze(1)).view(bs, -1)
            
            res = self.res_block(cross_fusion)
            res_concat = torch.cat([cross_fusion, res], dim=1)

            res2 = self.res_block2(res_concat)
            outputs = self.classifier(cross_fusion + res2)

            return outputs
        

        elif self.multimodal_method=='hybrid_fusion':
            bs = wav_vec.size()[0]
            
            # base
            base = torch.cat([wav_vec, rnn_vec], dim=1)
            base_outputs = self.base_classifier(self.base_norm(base))

            # late fusion
            wave_output = self.wave_classifier(wav_vec)
            text_output = self.text_classifier(rnn_vec)
            lf_outputs = torch.mean(self.lf_norm(torch.stack([wave_output, text_output], dim=1)), dim=1)
            
            # residuals
            x = self.res_norm(torch.cat([wav_vec, rnn_vec], dim=1))
            res = self.res_block(x)
            res_concat = torch.cat([x, res], dim=1)
            res2 = self.res_block2(res_concat)
            res_outputs = self.res_classifier(x + res2)

            # residuals_attn_cf
            wav_vec_proj = self.wav_proj(wav_vec)
            rnn_vec_proj = self.rnn_proj(rnn_vec)
            
            wav_vec = self.wav_norm(self.wave_attn(wav_vec_proj) + wav_vec_proj)
            rnn_vec = self.rnn_norm(self.text_attn(rnn_vec_proj) + rnn_vec_proj)
            
            _wav = torch.cat((
                Variable(torch.ones(bs, 1).type(wav_vec.dtype).to(wav_vec.device), requires_grad=False), wav_vec), dim=1)
            _rnn = torch.cat((
                Variable(torch.ones(bs, 1).type(rnn_vec.dtype).to(rnn_vec.device), requires_grad=False), rnn_vec), dim=1)
            
            cross_fusion = torch.matmul(_wav.unsqueeze(2), _rnn.unsqueeze(1)).view(bs, -1)
            
            res = self.rac_res_block(cross_fusion)
            res_concat = torch.cat([cross_fusion, res], dim=1)

            res2 = self.rac_res_block2(res_concat)
            rac_outputs = self.rac_classifier(cross_fusion + res2)
            
            # soft voting
            outputs = self.hybrid_norm(torch.stack([base_outputs, lf_outputs, res_outputs, rac_outputs], dim=1))
            outputs = torch.mean(outputs, dim=1)

        return outputs


class MultiModel(nn.Module):
    """wav2vec2, Roberta Multi Modal"""
    def __init__(self, config):
        super().__init__()

        self.wav_model = Wav2Vec2Model.from_pretrained(config.wav_model, config=config.wav_config)
        if 'xlm' in config.rnn_model:
            self.rnn_model = XLMRobertaModel.from_pretrained(config.rnn_model, config=config.rnn_config)
        else:
            from transformers import AutoModel
            self.rnn_model = AutoModel.from_pretrained(config.rnn_model, config=config.rnn_config)
        self.classifier = MultiModalClassificationHead(config)

    def wav_freeze_feature_extractor(self):
        self.wav_model.feature_extractor._freeze_parameters()

    def forward(self, input_values, input_ids, attention_mask, token_type_ids):

        wav_vec = torch.mean(
            self.wav_model(input_values)[0],
            dim=1)
        rnn_vec = torch.mean(
            self.rnn_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state'],
            dim=1)

        outputs = self.classifier(wav_vec, rnn_vec)

        return outputs