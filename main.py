import argparse
import importlib
from logging import debug
import numpy as np
import torch
import pytorch_lightning as pl
import lit_models
import yaml
import time
from transformers import AutoConfig
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# In order to ensure reproducible experiments, we must set random seeds.

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--max_triplet", type=int, default=32)
    parser.add_argument("--use_global_node", type=bool, default=False)
    parser.add_argument("--add_attn_bias", type=bool, default=True)
    # parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--chunk", type=str, default="")
    parser.add_argument("--model_class", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--checkpoint", type=str, default=None)



    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    lit_model_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    print('***********************')
    print(f"data.{temp_args.data_class}")
    print(f"models.{temp_args.model_class}")
    print(f"lit_models.{temp_args.litmodel_class}")
    print('***********************')

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    if hasattr(model_class, "add_to_argparse"):
        model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def _saved_pretrain(lit_model, tokenizer, path):
    lit_model.model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # update parameters
    config.label_smoothing = args.label_smoothing

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model)
    # print('data', data)
    tokenizer = data.tokenizer

    lit_model = litmodel_class(args=args, model=model, tokenizer=tokenizer, data_config=data.get_config())
    if args.checkpoint:
        lit_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"], strict=False)

    logger = pl.loggers.TensorBoardLogger("training/logs")
    # if args.wandb:
    #     logger = pl.loggers.WandbLogger(project="kgc_bert", name=args.data_dir.split("/")[-1])
    #     logger.log_hyperparams(vars(args))

    metric_name = "Eval/mrr" if not args.pretrain else "Eval/hits1"

    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/mrr", mode="max", patience=10)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor=metric_name, mode="max",
        filename=args.data_dir.split("/")[-1] + '/{epoch}-{Eval/hits10:.2f}-{Eval/hits1:.2f}' if not args.pretrain else args.data_dir.split("/")[-1] + '/{epoch}-{step}-{Eval/hits10:.2f}',
        dirpath="output",
        save_weights_only=True,
        every_n_train_steps=100 if args.pretrain else None,
        # every_n_train_steps=100,
        save_top_k=5 if args.pretrain else 1
    )
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs")
    
    if "EntityEmbedding" not in lit_model.__class__.__name__:
        trainer.fit(lit_model, datamodule=data)
        path = model_checkpoint.best_model_path        
        lit_model.load_state_dict(torch.load(path)["state_dict"], strict=False)

    result = trainer.test(lit_model, datamodule=data)
    print(result)

    # _saved_pretrain(lit_model, tokenizer, path)
    # if "EntityEmbedding" not in lit_model.__class__.__name__:
    #     print("*path"*30)
    #     print(path)

if __name__ == "__main__":
    main()
