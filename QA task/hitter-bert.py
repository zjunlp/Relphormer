import json
import math
import argparse
from pathlib import Path

from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup, AutoConfig

import torch
from torch import device, nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from transformers.tokenization_bert import BertTokenizerFast

from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.util import sc
# from relphormer.lit_models import TransformerLitModel
from relphormer.models import BertKGC
# from relphormer.data import KGC
import os

os.environ['CUDA_VISIBLE_DEVICES']='4'

MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL)


class FBQADataset(Dataset):

    def __init__(self, file_dir):
        self.examples = json.load(Path(file_dir).open("rb"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.examples[idx]


def fbqa_collate(samples):
    questions = []
    answers = []
    answer_ids = []
    entities = []
    entity_names = []
    relations = []
    for item in samples:
        q = item["RawQuestion"] + "[MASK]" * len(item["AnswerEntity"]) + "."
        questions.append(q)
        answers.append(item["AnswerEntity"])
        answer_ids.append(item["AnswerEntityID"])
        entities.append(item["TopicEntityID"])
        entity_names.append(item["TopicEntityName"])
        relations.append(item["RelationID"])
        
    questions = tokenizer(questions, return_tensors='pt', padding=True)
    entity_names = tokenizer(entity_names, add_special_tokens=False)
    answers, answers_lengths = sc.pad_seq_of_seq(answers)
    answers = torch.LongTensor(answers)
    answers_lengths = torch.LongTensor(answers_lengths)
    answer_ids = torch.LongTensor(answer_ids)

    input_ids = questions['input_ids']
    masked_labels = torch.ones_like(input_ids) * -100
    masked_labels[input_ids == tokenizer.mask_token_id] = answers[answers != 0]
    entity_mask = torch.zeros_like(input_ids).bool()
    entity_span_index = input_ids.new_zeros((len(input_ids), 2))
    for i, e_tokens in enumerate(entity_names['input_ids']):
        q_tokens = input_ids[i].tolist()
        for s_index in range(len(q_tokens) - len(e_tokens)):
            if all([e_token == q_tokens[s_index + j] for j, e_token in enumerate(e_tokens)]):
                entity_mask[i][s_index:s_index + len(e_tokens)] = True
                entity_span_index[i][0] = s_index
                entity_span_index[i][1] = s_index + len(e_tokens) - 1
                break

    entities = torch.LongTensor(entities)
    relations = torch.LongTensor(relations)

    return questions.data, masked_labels, answers, answers_lengths, answer_ids, entities, relations, entity_mask, entity_span_index


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CrossAttention(nn.Module):
    def __init__(self, config, ctx_hidden_size):
        super().__init__()
        self.self = CrossAttentionInternal(config, ctx_hidden_size)
        self.output = SelfOutput(config)
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CrossAttentionInternal(nn.Module):
    def __init__(self, config, ctx_hidden_size):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_hidden_size, self.all_head_size)
        self.value = nn.Linear(ctx_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        mixed_key_layer = self.key(encoder_hidden_states)
        mixed_value_layer = self.value(encoder_hidden_states)
        attention_mask = encoder_attention_mask

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, nn.Softmax(dim=-1)(attention_scores)) if output_attentions else (context_layer,)
        return outputs


class CrossTrmFinetuner(pl.LightningModule):
    def __init__(self, hparams, bertmodel):
        super().__init__()
        self._hparams = hparams

        self.lr = hparams['lr']
        self.weight_decay = hparams['weight_decay']

        self.kg_dim = 320
        # self.bert = BertForMaskedLM.from_pretrained(MODEL)
        self.bert = bertmodel

        if self._hparams['use_hitter']:
            self.kg_layer_num = 10
            self.cross_attentions = nn.ModuleList([CrossAttention(self.bert.config, self.kg_dim)
                                                   for _ in range(self.kg_layer_num)])
            checkpoint = load_checkpoint('local/best/20200812-174221-trmeh-fb15k237-best/checkpoint_best.pt')
            self.hitter = KgeModel.create_from(checkpoint)

    def forward(self, batch):
        sent_input, masked_labels, batch_labels, label_lens, answer_ids, s, p, entity_mask, entity_span_index = batch

        if self._hparams['use_hitter']:
            # kg_masks: [bs, 1, 1, length]
            # kg_embeds: nlayer*[bs, length, dim]
            kg_embeds, kg_masks = self.hitter('get_hitter_repr', s, p)
            kg_attentions = [None] * 2 + [(self.cross_attentions[i], kg_embeds[(i + 2) // 2], kg_masks)
                                          for i in range(self.kg_layer_num)]
        else:
            kg_attentions = []

        out = self.bert(kg_attentions=kg_attentions,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                        labels=masked_labels,
                        **sent_input,
                        )

        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        batch_inputs, masked_labels, batch_labels, label_lens, answer_ids, s, p, entity_mask, _ = batch
        output = self(batch)
        input_tokens = batch_inputs["input_ids"].clone()

        logits = output.logits[masked_labels != -100]
        probs = logits.softmax(dim=-1)
        values, predictions = probs.topk(1)
        hits = []
        now_pos = 0
        for sample_i, label_length in enumerate(label_lens.tolist()):
            failed = False
            for i in range(label_length):
                if (predictions[now_pos + i] == batch_labels[sample_i][i]).sum() != 1:
                    failed = True
                    break
            hits += [1] if not failed else [0]
            now_pos += label_length
        hits = torch.tensor(hits)
        input_tokens[input_tokens == tokenizer.mask_token_id] = predictions.flatten()
        pred_strings = [str(hits[i].item()) + ' ' + tokenizer.decode(input_tokens[i], skip_special_tokens=True)
                        for i in range(input_tokens.size(0))]

        return {'val_loss': output.loss,
                'val_acc': hits.float(),
                'pred_strings': pred_strings}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.cat([x['val_acc'] for x in outputs]).mean().to(avg_loss.device)

        if self.global_rank == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_text('pred', '\n\n'.join(sum([x['pred_strings'] for x in outputs], [])), self.global_step)

        self.log('avg_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': avg_loss}

    def train_dataloader(self):
        return DataLoader(FBQADataset(self._hparams['train_dataset']),
                          self._hparams['batch_size'],
                          shuffle=True,
                          collate_fn=fbqa_collate,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(FBQADataset(self._hparams['val_dataset']),
                          1,
                          shuffle=False,
                          collate_fn=fbqa_collate,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(FBQADataset(self._hparams['test_dataset']),
                          1,
                          shuffle=False,
                          collate_fn=fbqa_collate,
                          num_workers=0)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        no_fine_tune = ['cross_attentions']
        pgs = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and not any([i in n for i in no_fine_tune])],
                'weight_decay': 0.01},
               {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and not any([i in n for i in no_fine_tune])],
                'weight_decay': 0.0}]
        if self._hparams['use_hitter']:
            pgs.append({'params': self.cross_attentions.parameters(), 'lr': 5e-5, 'weight_decay': 0.01})
        # bert_optimizer = AdamW(pgs, lr=3e-5, weight_decay=1e-2)
        bert_optimizer = AdamW(pgs, lr=self.lr, weight_decay=self.weight_decay)
        bert_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(bert_optimizer, self._hparams['max_steps'] // 10, self._hparams['max_steps']),
            'interval': 'step',
            'monitor': None
        }
        return [bert_optimizer], [bert_scheduler]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='default', nargs='?', help="Name of the experiment")
    parser.add_argument('--dataset', choices=['fbqa', 'webqsp'], default='fbqa', help="fbqa or webqsp")
    parser.add_argument('--filtered', default=False, action='store_true', help="Filtered or not")
    parser.add_argument('--hitter', default=False, action='store_true', help="Use pretrained HittER or not")
    parser.add_argument('--relphormer', default=False, action='store_true', help="Use pretrained relphormer or not")
    parser.add_argument('--seed', default=333, type=int, help='Seed number')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
    args = parser.parse_args()
    seed_everything(args.seed)

    QA_DATASET = args.dataset
    if args.filtered and args.relphormer:
        SUBSET = 'relphormer-filtered'
    elif not args.filtered and args.relphormer:
        SUBSET = 'relphormer'
    elif args.filtered and not args.relphormer:
        SUBSET = 'fb15k237-filtered'
    else:
        SUBSET = 'fb15k237'

    hparams = {
        'use_hitter': args.hitter,
        'relphormer': args.relphormer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': 16,
        'max_epochs': 20,
        'train_dataset': f'data/{QA_DATASET}/{SUBSET}/train.json',
        'val_dataset': f'data/{QA_DATASET}/{SUBSET}/test.json',
        'test_dataset': f'data/{QA_DATASET}/{SUBSET}/test.json',
    }

    if hparams['relphormer']:
        MODEL = "./local/relphormer/"
        config = AutoConfig.from_pretrained(MODEL)
        bertmodel = BertForMaskedLM.from_pretrained(MODEL, config=config)
        model = CrossTrmFinetuner(hparams, bertmodel=bertmodel)
    else:
        bertmodel = BertForMaskedLM.from_pretrained(MODEL)
        model = CrossTrmFinetuner(hparams, bertmodel=bertmodel)
    model.hparams['max_steps'] = (len(model.train_dataloader().dataset) // hparams['batch_size'] + 1) * hparams['max_epochs']
    base_path = '/tmp/hitbert-paper'
    logger = TensorBoardLogger(base_path, args.exp_name)
    checkpoint_callback = ModelCheckpoint(
            monitor='avg_val_acc',
            dirpath=base_path + '/' + args.exp_name,
            filename='{epoch:02d}-{avg_val_acc:.3f}',
            save_top_k=1,
            mode='max')
    trainer = pl.Trainer(gpus=1, accelerator="ddp",
                         max_epochs=hparams['max_epochs'], max_steps=model.hparams['max_steps'],
                         checkpoint_callback=True,
                         gradient_clip_val=1.0, logger=logger,
                         callbacks=[LearningRateMonitor(), checkpoint_callback])
    trainer.fit(model)
    print("QA Task End!")
