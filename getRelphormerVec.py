import argparse
from transformers import AutoTokenizer, AutoConfig
import torch


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path")
    parser.add_argument("--checkpoint")

    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config =  AutoConfig.from_pretrained(args.model_name_or_path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    vocab_size = config.vocab_size
    word_embeddings = checkpoint['state_dict']['model.bert.embeddings.word_embeddings.weight']
    add_size = list(word_embeddings.shape)[0]

    KG_embedding = word_embeddings[vocab_size: add_size, ]
    torch.save(KG_embedding, 'KG_embedding.pt')



if __name__ == "__main__":
    main()
