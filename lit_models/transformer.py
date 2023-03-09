from logging import debug
import random
from turtle import distance
import pytorch_lightning as pl
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from transformers.utils.dummy_pt_objects import PrefixConstrainedLogitsProcessor

from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from functools import partial
from .utils import rank_score, acc, LabelSmoothSoftmaxCEV1

from typing import Callable, Iterable, List

def pad_distance(pad_length, distance):
    pad = nn.ConstantPad2d(padding=(0, pad_length, 0, pad_length), value=float('-inf'))
    distance = pad(distance)
    return distance

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        if args.bce:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.first = True
        
        self.tokenizer = tokenizer
        self.num_heads = 12
        self.__dict__.update(data_config)
        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        if args.pretrain:
            self._freaze_attention()
        elif "ind" in args.data_dir:
            # for inductive setting, use feeaze the word embedding
            self._freaze_word_embedding()
        
        self.graph_token_virtual_distance = nn.Embedding(1, self.num_heads)
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # if batch_idx == 0:
        #     print('\n'.join(self.decode(batch['input_ids'][:1])))

        labels = batch.pop("labels")
        label = batch.pop("label")
        
        input_ids = batch['input_ids']

        # edited by bizhen
        distance_attention = batch.pop("distance_attention")
        distance_attention = distance_attention
        
        graph_attn_bias = torch.zeros(input_ids.size(0), input_ids.size(1), input_ids.size(1)).cuda()
        graph_attn_bias[:, 1:, 1:][distance_attention == float('-inf')] = float('-inf')
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + distance_attention.unsqueeze(1)
        
        if self.args.use_global_node:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        # edited by bizhen

        if self.args.add_attn_bias:
            logits = self.model(**batch, return_dict=True, distance_attention=graph_attn_bias).logits
        else:
            logits = self.model(**batch, return_dict=True, distance_attention=None).logits


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_logits = logits[torch.arange(bs), mask_idx][:, self.entity_id_st:self.entity_id_ed]

        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        if self.args.bce:
            loss = self.loss_fn(mask_logits, labels)
        else:
            loss = self.loss_fn(mask_logits, label)

        return loss

    def _eval(self, batch, batch_idx):
        # single label
        labels = batch.pop("labels")    
        label = batch.pop('label')
        
        input_ids = batch['input_ids']
        
        # edited by bizhen
        distance_attention = batch.pop("distance_attention")
        distance_attention = distance_attention
        
        graph_attn_bias = torch.zeros(input_ids.size(0), input_ids.size(1), input_ids.size(1)).cuda()
        graph_attn_bias[:, 1:, 1:][distance_attention == float('-inf')] = float('-inf')
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + distance_attention.unsqueeze(1)
        
        if self.args.use_global_node:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        if self.args.add_attn_bias:
            logits = self.model(**batch, return_dict=True, distance_attention=graph_attn_bias).logits[:, :, self.entity_id_st:self.entity_id_ed]
        else:
            logits = self.model(**batch, return_dict=True, distance_attention=None).logits[:, :, self.entity_id_st:self.entity_id_ed]
            
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx]
        # get the entity ranks
        # filter the entity
        assert labels[0][label[0]], "correct ids must in filiter!"
        labels[torch.arange(bsz), label] = 0
        assert logits.shape == labels.shape
        logits += labels * -100 # mask entityj
        # for i in range(bsz):
        #     logits[i][labels]

        _, outputs = torch.sort(logits, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        

        return dict(ranks = np.array(ranks))

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])
        total_ranks = ranks.shape[0]

        if not self.args.pretrain:
            l_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2)))]
            r_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2))) + 1]
            self.log("Eval/lhits10", (l_ranks<=10).mean())
            self.log("Eval/rhits10", (r_ranks<=10).mean())

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10)
        self.log("Eval/hits20", hits20)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits1", hits1)
        self.log("Eval/mean_rank", ranks.mean())
        self.log("Eval/mrr", (1. / ranks).mean())
        self.log("hits10", hits10, prog_bar=True)
        self.log("hits1", hits1, prog_bar=True)
   

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # ranks = self._eval(batch, batch_idx)
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))

        return result

    def test_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Test/hits10", hits10)
        self.log("Test/hits20", hits20)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mean_rank", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer_class(self.parameters(), lr=self.lr, eps=1e-8)
        }
    
    def _freaze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freaze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
