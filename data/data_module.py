from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from enum import Enum
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
# from transformers.configuration_bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

from .base_data_module import BaseDataModule
from .processor import KGProcessor, get_dataset
import transformers
transformers.logging.set_verbosity_error()

import torch.nn as nn

class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"

import numpy as np

def pad_distance(pad_length, distance):
    pad = nn.ConstantPad2d(padding=(0, pad_length, 0, pad_length), value=float('-inf'))
    distance = pad(distance)
    return distance

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None
        label = [feature.pop("label") for feature in features]
        features_keys = {}
        name_keys = list(features[0].keys())
        for k in name_keys:
            # ignore the padding arguments
            if k in ["input_ids", "attention_mask", "token_type_ids"]: 
                continue
            try:
                features_keys[k] = [feature.pop(k) for feature in features]
            except KeyError:
                print('error!')
                continue
                

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        bsz = len(labels)
        with torch.no_grad():
            new_labels = torch.zeros(bsz, self.num_labels)
            for i,l in enumerate(labels):
                if isinstance(l, int): 
                    new_labels[i][l] = 1
                else:
                    for j in l:
                        new_labels[i][j] = 1
            labels = new_labels

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features['labels'] = labels
        features['label'] = torch.tensor(label)
        features.update(features_keys)
        
        # editted by bizhen
        features['distance_attention'] = torch.stack([pad_distance(len(features['input_ids'][i]) - len(distance) - 1, distance) for i, distance in enumerate(features['distance_attention'])])

        return features



class KGC(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        self.processor = KGProcessor(self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)

        entity_list = self.processor.get_entities(args.data_dir)
        
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        self.sampler = DataCollatorForSeq2Seq(self.tokenizer,
            model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.args.precision == 16 else None,
            padding="longest",
            max_length=self.args.max_seq_length,
            num_labels = len(entity_list),
        )
        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relations_tokens})

        vocab = self.tokenizer.get_added_vocab()
        self.relation_id_st = vocab[relations_tokens[0]]
        self.relation_id_ed = vocab[relations_tokens[-1]] + 1
        self.entity_id_st = vocab[entity_list[0]]
        self.entity_id_ed = vocab[entity_list[-1]] + 1


    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "train")
        self.data_val = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "dev")
        self.data_test = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "test")

    def prepare_data(self):
        pass

    def get_config(self):
        d = {}
        for k, v in self.__dict__.items():
            if "st" in k or "ed" in k:
                d.update({k:v})
        
        return d


    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--data_dir", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=self.num_workers,  collate_fn=self.sampler, batch_size=self.args.batch_size, shuffle=not self.args.faiss_init)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=self.num_workers, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=self.num_workers, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)