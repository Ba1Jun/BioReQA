import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tqdm import tqdm



class RankModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type

        if self.encoder_type == 'biobert':
            self.encoder = BertModel.from_pretrained(args.plm_path)

        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.softmax = nn.Softmax(dim=1)
    

    def pooler(self, last_hidden_state, attention_mask):
        if self.pooler_type == 'mean':
            pooler_output = torch.sum(last_hidden_state * attention_mask.unsqueeze(2), dim=1)
            pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True)
        return pooler_output


    def sentence_encoding(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, 
                                        attention_mask=attention_mask,
                                        return_dict=True)
        last_hidden_state = encoder_outputs['last_hidden_state']

        pooler_output = self.pooler(last_hidden_state, attention_mask)
        pooler_output = F.normalize(pooler_output, dim=-1)
        return pooler_output

    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, src_ids=None, inference=False):
        # [bs, d_model]
        src_pooler_output = self.sentence_encoding(src_input_ids, src_attention_mask)
        tgt_pooler_output = self.sentence_encoding(tgt_input_ids, tgt_attention_mask)
 
        if not inference:
            # [bs, bs]
            predict_logits = src_pooler_output.mm(tgt_pooler_output.t())
            predict_logits /= self.temperature
            # loss
            if src_ids is not None:
                batch_size = src_pooler_output.shape[0]
                logit_mask = (src_ids.unsqueeze(1).repeat(1, batch_size) == src_ids.unsqueeze(0).repeat(batch_size, 1)).float() - torch.eye(batch_size).to(src_ids.device)
                predict_logits -= logit_mask * 100000000
            label = torch.arange(0, predict_logits.shape[0]).to(src_input_ids.device)
            predict_loss = self.ce_loss(predict_logits, label)
                
            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            return predict_loss, acc
        else:
            return src_pooler_output, tgt_pooler_output