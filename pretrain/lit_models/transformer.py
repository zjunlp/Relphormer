from logging import debug
import random
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

        self.__dict__.update(data_config)
        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        if args.pretrain:
            self._freaze_attention()
        elif "ind" in args.data_dir:
            # for inductive setting, use feeaze the word embedding
            self._freaze_word_embedding()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        # embed();exit()
        # print(self.optimizers().param_groups[1]['lr'])
        labels = batch.pop("labels")
        label = batch.pop("label")
        pos = batch.pop("pos")
        try:
            en = batch.pop("en")
            rel = batch.pop("rel")
        except KeyError:
            pass
        input_ids = batch['input_ids']
        logits = self.model(**batch, return_dict=True).logits

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_logits = logits[torch.arange(bs), mask_idx][:, self.entity_id_st:self.entity_id_ed]

        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        if self.args.bce:
            loss = self.loss_fn(mask_logits, labels)
        else:
            loss = self.loss_fn(mask_logits, label)

        if batch_idx == 0:
            print('\n'.join(self.decode(batch['input_ids'][:4])))
        

        return loss

    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')
        pos = batch.pop('pos')
        my_keys = list(batch.keys())
        for k in my_keys:
            if k not in ["input_ids", "attention_mask", "token_type_ids"]:
                batch.pop(k)
        logits = self.model(**batch, return_dict=True).logits[:, :, self.entity_id_st:self.entity_id_ed]
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
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
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



import faiss
import os
class GetEntityEmbeddingLitModel(TransformerLitModel):
    def __init__(self, model, args, tokenizer, data_config={}):
        super().__init__(model, args, tokenizer, data_config)

        self.faissid2entityid = {}
        # self.index = faiss.IndexFlatL2(d)   # build the index

        d, measure = self.model.config.hidden_size, faiss.METRIC_L2   
        # param =  'HNSW64' 
        # self.index = faiss.index_factory(d, param, measure)  
        self.index = faiss.IndexFlatL2(d)   # build the index
        # print(self.index.is_trained)                          # 此时输出为True 
        # index.add(xb)
        self.cnt_batch = 0
        self.total_embedding = []



    def test_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        mask_idx = batch.pop("pos")
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')
        # last layer 
        hidden_states = self.model(**batch, return_dict=True, output_hidden_states=True).hidden_states[-1]
        # _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        entity_embedding = hidden_states[torch.arange(bsz), mask_idx].cpu()
        # use normalize or not ?
        # entity_embedding = F.normalize(entity_embedding, dim=-1, p = 2)
        self.total_embedding.append(entity_embedding)
        # self.index.add(np.array(entity_embedding, dtype=np.float32))
        for i, l in zip(range(bsz), label):
            self.faissid2entityid[i+self.cnt_batch] = l.cpu()
        self.cnt_batch += bsz


    def test_epoch_end(self, outputs) -> None:
        self.total_embedding = np.concatenate(self.total_embedding, axis=0)
        # self.index.train(self.total_embedding)
        print(faiss.MatrixStats(self.total_embedding).comments)
        self.index.add(self.total_embedding)
        faiss.write_index(self.index, os.path.join(self.args.data_dir, "faiss_dump.index"))
        with open(os.path.join(self.args.data_dir, "faissid2entityid.pkl") ,'wb') as file:
            pickle.dump(self.faissid2entityid, file)

        with open(os.path.join(self.args.data_dir, "total_embedding.pkl") ,'wb') as file:
            pickle.dump(self.total_embedding, file)
        # print(f"number of  entity embedding : {len(self.faissid2entityid)}")

    @staticmethod
    def add_to_argparse(parser):
        parser = TransformerLitModel.add_to_argparse(parser)
        parser.add_argument("--faiss_init", type=int, default=1, help="get the embedding and save it the file.")
        return parser

class UseEntityEmbeddingLitModel(TransformerLitModel):
    def __init__(self, model, args, tokenizer, data_config={}):
        super().__init__(model, args, tokenizer, data_config)

        self.faissid2entityid = pickle.load(open(os.path.join(self.args.data_dir, "faissid2entityid.pkl") ,'rb'))
        self.index = faiss.read_index(os.path.join(self.args.data_dir, "faiss_dump.index"))
        

        self.dis2logits = distance2logits_2
    
    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        pos = batch.pop("pos")
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')

        hidden_states = self.model(**batch, return_dict=True, output_hidden_states=True).hidden_states[-1]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        mask_embedding = np.array(hidden_states[torch.arange(bsz), mask_idx].cpu(), dtype=np.float32)
        topk = 200
        D, I = self.index.search(mask_embedding, topk)
        labels[torch.arange(bsz), label] = 0

        # logits = torch.zeros_like(labels)
        # D = torch.softmax(torch.exp(-1. * torch.tensor(D)), dim=-1)
        # for i in range(bsz):
        #     for j in range(topk):
        #         logits[i][self.faissid2entityid[I[i][j]]] += D[i][j]
        # # get the entity ranks
        # # filter the entity
        # assert labels[0][label[0]], "correct ids must in filiter!"
        # labels[torch.arange(bsz), label] = 0
        # assert logits.shape == labels.shape
        # logits += labels * -100 # mask entityj

        # _, outputs = torch.sort(logits, dim=1, descending=True)
        # _, outputs = torch.sort(outputs, dim=1)
        # ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1


        entity_logits = torch.full(labels.shape, -100.).to(self.device)
        D = self.dis2logits(D)
        for i in range(bsz):
            for j in range(topk):
                # filter entity in labels
                if I[i][j] not in self.faissid2entityid: 
                    print(I[i][j])
                    break
                # assert I[i][j] in self.faissid2entityid, print(I[i][j])
                if labels[i][self.faissid2entityid[I[i][j]]]: continue
                if entity_logits[i][self.faissid2entityid[I[i][j]]] == -100.:
                    entity_logits[i][self.faissid2entityid[I[i][j]]] = D[i][j]
                # no added together
                # else:
                #     entity_logits[i][self.faissid2entityid[I[i][j]]] += D[i][j]
        # get the entity ranks
        # filter the entity

        assert entity_logits.shape == labels.shape

        _, outputs = torch.sort(entity_logits, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        

        return dict(ranks = np.array(ranks))
    

    @staticmethod
    def add_to_argparse(parser):
        parser = TransformerLitModel.add_to_argparse(parser)
        parser.add_argument("--faiss_init", type=int, default=0, help="get the embedding and save it the file.")
        parser.add_argument("--faiss_use", type=int, default=1, help="get the embedding and save it the file.")
        return parser


class CombineEntityEmbeddingLitModel(UseEntityEmbeddingLitModel):
    def __init__(self, model, args, tokenizer, data_config={}):
        super().__init__(model, args, tokenizer, data_config=data_config)
        self.dis2logits = distance2logits_2
        self.id2entity = {}
        with open("./dataset/FB15k-237/entity2textlong.txt", 'r') as file:
            cnt = 0
            for line in file.readlines():
                e, d = line.strip().split("\t")
                self.id2entity[cnt] = e
                cnt += 1
        self.id2entity_t = {}
        with open("./dataset/FB15k-237/entity2text.txt", 'r') as file:
            for line in file.readlines():
                e, d = line.strip().split("\t")
                self.id2entity_t[e] = d
        for k, v in self.id2entity.items():
            self.id2entity[k] = self.id2entity_t[v]
    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')
        pos = batch.pop("pos")

        result = self.model(**batch, return_dict=True, output_hidden_states=True)
        hidden_states = result.hidden_states[-1]
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bsz = input_ids.shape[0]
        mask_embedding = np.array(hidden_states[torch.arange(bsz), mask_idx].cpu(), dtype=np.float32)
        # mask_embedding = np.array(hidden_states[torch.arange(bsz), mask_idx].cpu(), dtype=np.float32)
        topk = self.args.knn_topk
        D, I = self.index.search(mask_embedding, topk)
        D = torch.from_numpy(D).to(self.device)
        assert labels[0][label[0]], "correct ids must in filiter!"
        labels[torch.arange(bsz), label] = 0


        mask_logits = result.logits[:, :, self.entity_id_st:self.entity_id_ed]
        mask_logits = mask_logits[torch.arange(bsz), mask_idx]
        entity_logits = torch.full(labels.shape, 1000.).to(self.device)
        # D = self.dis2logits(D)
        for i in range(bsz):
            for j in range(topk):
                # filter entity in labels
                if labels[i][self.faissid2entityid[I[i][j]]]: continue
                if entity_logits[i][self.faissid2entityid[I[i][j]]] == 1000.:
                    entity_logits[i][self.faissid2entityid[I[i][j]]] = D[i][j]
                # else:
                #     entity_logits[i][self.faissid2entityid[I[i][j]]] += D[i][j]
        entity_logits = self.dis2logits(entity_logits)
        # get the entity ranks
        # filter the entity
        assert entity_logits.shape == labels.shape
        assert mask_logits.shape == labels.shape
        # entity_logits = torch.softmax(entity_logits + labels * -100, dim=-1) # mask entityj
        entity_logits = entity_logits + labels* -100.
        mask_logits = torch.softmax(mask_logits + labels* -100, dim=-1)
        # logits = mask_logits
        logits = combine_knn_and_vocab_probs(entity_logits, mask_logits, self.args.knn_lambda)
        # logits = entity_logits + mask_logits


        knn_topk_logits, knn_topk_id  = entity_logits.topk(20)
        mask_topk_logits, mask_topk_id  = mask_logits.topk(20)
        union_topk = []
        for i in range(bsz):
            num_same = len(list(set(knn_topk_id[i].cpu().tolist()) & set(mask_topk_id[i].cpu().tolist())))
            union_topk.append(num_same/ 20.)
        
        knn_topk_id = knn_topk_id.to("cpu")
        mask_topk_id = mask_topk_id.to("cpu")
        mask_topk_logits = mask_topk_logits.to("cpu")
        knn_topk_logits = knn_topk_logits.to("cpu")
        label = label.to("cpu")



        for t in range(bsz):
            if knn_topk_id[t][0] == label[t] and knn_topk_logits[t][0] > mask_topk_logits[t][0] and mask_topk_logits[t][0] <= 0.4:
                print(knn_topk_logits[t], knn_topk_id[t])
                print(lmap(lambda x: self.id2entity[x.item()], knn_topk_id[t]))
                print(mask_topk_logits[t], mask_topk_id[t])
                print(lmap(lambda x: self.id2entity[x.item()], mask_topk_id[t]))
                print(label[t])
                print()

        _, outputs = torch.sort(logits, dim=1, descending=True)
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        

        return dict(ranks = np.array(ranks), knn_topk_id=knn_topk_id, knn_topk_logits=knn_topk_logits,
            mask_topk_id=mask_topk_id, mask_topk_logits=mask_topk_logits, num_same = np.array(union_topk))
    
    def test_epoch_end(self, outputs) -> None:

        ranks = np.concatenate([_['ranks'] for _ in outputs])
        num_same = np.concatenate([_['num_same'] for _ in outputs])
        results_keys = list(outputs[0].keys())
        results = {}
        # for k in results_keys:
        #     results.

        self.log("Test/num_same", num_same.mean())

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

    def add_to_argparse(parser):
        parser = TransformerLitModel.add_to_argparse(parser)
        parser.add_argument("--knn_lambda", type=float, default=0.5, help="lambda * knn + (1-lambda) * mask logits , lambda of knn logits and mask logits.")
        parser.add_argument("--knn_topk", type=int, default=100, help="")

        return parser

def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff=0.5):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob

def distance2logits(D):
    return torch.softmax( -1. * torch.tensor(D) / 30., dim=-1)

def distance2logits_2(D, n=10):
    if not isinstance(D, torch.Tensor):
        D = torch.tensor(D)
    if torch.sum(D) != 0.0:
        distances = torch.exp(-D/n) / torch.sum(torch.exp(-D/n), dim=-1, keepdim=True)
    return distances