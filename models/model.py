from transformers.models.bert.modeling_bert import BertForMaskedLM


class BertKGC(BertForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser


from models.huggingface_relformer import BertForMaskedLM as BertForMaskedLM_
class Relphormer(BertForMaskedLM_):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrain", type=int, default=0, help="")
        return parser
