import os
import typing
import time
import random
import re

import argparse
import logging
import json

import numpy as np

import pandas as pd

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset as __Dataset__, DataLoader
from torch.optim import Optimizer, AdamW

from transformers import AutoTokenizer
from src import T5VAEForConditionalGeneration


random.seed(42)
torch.manual_seed(42)

def parse_args() -> typing.Dict:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_version", type=str, required=True, help="Movel version")
    parser.add_argument("--tokenizer_config", type=str, default="ElnaggarLab/ankh-base", help="Path or Huggingface's repository of the model's tokenizer")
    parser.add_argument("--from_model", type=str, help="Path to repository containing the model's encoder and decoder")
    parser.add_argument("--vae_parameters", type=str, help="[512, 256, 128];64;0.2;32")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--d_learning_rate", type=float, help="Learning rate")
    parser.add_argument("--earlystop_mode", type=str, help="Earlystops'statistic: [None|mean|median|quantile<integer between 0 and 100>]", default=None)
    parser.add_argument("--warmup", type=int, help="Numer of warmup steps")
    parser.add_argument("--patience", type=int, help="Number of patience steps")
    parser.add_argument("--training_set", type=str, required=True, help="Path to training set")
    parser.add_argument("--validation_set", type=str, required=True, help="Path to validation set")
    parser.add_argument("--min_length", type=int, required=True, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--n_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Mini-Batch size")
    parser.add_argument("--f_model", type=float, default=1)
    parser.add_argument("--f_recon", type=float, default=1)
    parser.add_argument("--f_kld", type=float, default=1)
    parser.add_argument("--f_adv", type=float, default=1)
    parser.add_argument("--f_var", type=float, default=1)
    parser.add_argument("--teacher_forcing", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=10, help="Number of sub-processes to use for data loading.")
    args = vars(parser.parse_args())

    dims_hidden, dim_latent, p, dim_discriminator = args["vae_parameters"].replace(" ", "").split(";")
    args["dims_hidden"] = [int(el) for el in dims_hidden[1:-1].split(",")]
    args["dim_latent"] = int(dim_latent)
    args["p"] = float(p)
    args["dim_discriminator"] = int(dim_discriminator)

    if not args["batch_size"] % args["mini_batch_size"] == 0:
        raise ValueError(f"--batch_size ({args['batch_size']}) should be a multiple of --mini_batch_size ({args['mini_batch_size']})")

    if args["earlystop_mode"] is None:
        args["warmup"] = 0
        args["patience"] = args["n_epochs"]

    if not os.path.isdir(args["model_dir"]):
        os.mkdir(args["model_dir"])

    if not os.path.isdir(os.path.join(args["model_dir"], args["model_name"])):
        os.mkdir(os.path.join(args["model_dir"], args["model_name"]))

    if os.path.isdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"])):
        raise FileExistsError("The same version of the model exists, please choose a new version")

    os.mkdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"]))
    os.mkdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best"))
    os.mkdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final"))

    if not os.path.isdir(args["log_dir"]):
        os.mkdir(args["log_dir"])

    if not os.path.isdir(os.path.join(args["log_dir"], args["model_name"])):
        os.mkdir(os.path.join(args["log_dir"], args["model_name"]))

    logging.basicConfig(filename=os.path.join(args["log_dir"], args["model_name"], args["model_version"] + ".log"), level=logging.INFO)

    with open(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "args.json"), 'w', encoding='utf8') as f:
        f.write(json.dumps(args, indent=4, sort_keys=False, separators=(',', ': '), ensure_ascii=False))

    return args

class Dataset(__Dataset__):

    def __init__(self, path, min_length, max_length):
        if not os.path.isfile(path):
            raise FileNotFoundError()
        
        self.df = pd.read_csv(path, header=None, sep=",")

        if not self.df.shape[1] == 1:
            raise NotImplementedError()
        
        self.df.columns = ["seq"]
        self.df = self.df[self.df.apply(lambda row: min_length < len(row["seq"]) < max_length, axis=1)].reset_index(drop=True)
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(
            self, 
            index: int
        ) -> str:
        protein = self.df.loc[index, "seq"]
        fragment = protein[(start:=random.randint(0, len(protein)//3)):random.randint(start+len(protein)//2, len(protein))]
        return fragment

def get_collate_fn(
        tokenizer: AutoTokenizer
    ) -> typing.Callable:

    def collate_fn(
            samples: typing.List[str]
        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        tokens = tokenizer(samples, padding=True, return_tensors="pt")
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        return input_ids, attention_mask
    
    return collate_fn

class EarlyStop():

    def __init__(
        self,
        patience: int,
        warmup: int,
        state: int=1,
        best_val_loss: float=float("inf"),
        best_epoch: int=0
    ) -> None:

        self.patience = patience
        self.warmup = warmup
        self.state = state
        self.best_val_loss = best_val_loss
        self.best_epoch = best_epoch

    def __str__(self) -> str:

        return f"[Best validation loss ({self.best_val_loss:.4f}) at epoch {self.best_epoch} | State: {self.state}/{self.patience}]"

    def __call__(
        self, 
        val_loss: float, 
        i_epoch: int
    ) -> typing.Tuple[bool, bool]:

        if val_loss < self.best_val_loss * 0.98:
            self.best_val_loss = float(val_loss)
            self.best_epoch = i_epoch
            self.state = 1
            return False, True
        
        if i_epoch <= self.warmup:
            return False, False

        self.state += 1

        if self.state >= self.patience:
            return True, False
        
        return False, False

    def save(
        self, 
        path: typing.Union[str, os.PathLike]
    ) -> None:

        with open(path, 'w', encoding='utf8') as f:
            f.write(json.dumps(self.__dict__, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False))

    def load(
        path: typing.Union[str, os.PathLike]
    ):

        return EarlyStop(**json.load(path))

def train(
    tokenizer: AutoTokenizer,
    model: T5VAEForConditionalGeneration,
    discriminator: nn.Module,
    optimizer: Optimizer,
    d_optimizer: Optimizer,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    collate_fn: typing.Callable,
    earlystop: EarlyStop,
    device: torch.DeviceObjType,
    args: typing.Dict
) -> None:

    _modes = ["train", "validation"]
    _metrics = ["model", "recon", "kld", "adv", "var", "total", "d"]
    _reductions = ["step", "mean", "median"]

    metrics = {f"{mode}/loss/{metric}/{reduction}":[] for mode in _modes for metric in _metrics for reduction in _reductions}

    model.to(device)
    model.to(torch.bfloat16)
    discriminator.to(device)
    discriminator.to(torch.bfloat16)

    accumulation_steps = args["batch_size"] // args["mini_batch_size"]

    for i_epoch in range(1, args["n_epochs"]+1):
        start_time = time.time()

        epoch_metrics = {f"{mode}/loss/{metric}" for mode in _modes for metric in _metrics}

        model.train()
        torch.cuda.empty_cache()

        for i_batch, batch in enumerate(training_dataloader):
            input_ids, attention_mask = [el.to(device) for el in batch]
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, teacher_forcing=args["teacher_forcing"])

            batch_metrics = {
                "model": out.loss,
                "recon": out.recon_loss,
                "kld": out.kld_loss,
                "var": out.var_loss
            }
            
            vae_latent_repr = out.vae_latent_repr.view(-1, out.vae_latent_repr.shape[-1])

            if torch.isnan(vae_latent_repr).any():
                raise ValueError()

            repr_means = np.mean(vae_latent_repr.float().cpu().detach().numpy(), axis=0)
            repr_cov = np.cov(vae_latent_repr.T.float().cpu().detach().numpy())
            fake = torch.tensor(np.random.multivariate_normal(repr_means, repr_cov, size=vae_latent_repr.shape[0]), device=device, dtype=torch.bfloat16)

            vae_latent_repr_preds = discriminator(vae_latent_repr)
            fake_preds = discriminator(fake)

            ce_loss_fn = CrossEntropyLoss()
            batch_metrics["adv"] = ce_loss_fn(vae_latent_repr_preds, torch.ones(vae_latent_repr_preds.shape[0], dtype=torch.long).to(device))
            batch_metrics["d"] = ce_loss_fn(vae_latent_repr_preds.detach(), torch.zeros(vae_latent_repr_preds.shape[0], dtype=torch.long).to(device)) + ce_loss_fn(fake_preds, torch.ones(fake_preds.shape[0], dtype=torch.long).to(device))

            batch_metrics["total"] = sum([args[f"f_{metric}"]*batch_metrics[metric] for metric in _metrics[:-2]])

            (batch_metrics["total"]/accumulation_steps).backward()
            (batch_metrics["d"] /accumulation_steps).backward()

            if (i_batch + 1) % accumulation_steps == 0 or i_batch + 1 == len(training_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                d_optimizer.step()
                d_optimizer.zero_grad()
            
            for metric in _metrics:
                metrics[f"train/loss/{metric}/step"].append(batch_metrics[{metric}].item())
                epoch_metrics[f"train/{metric}/model"].append(batch_metrics[{metric}].item())

            if i_batch % 100 == 0:
                torch.save(metrics, os.path.join(args["model_dir"], args["model_name"], args["model_version"], "metrics.bin"))

        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():

            for i_batch, batch in enumerate(validation_dataloader):
                input_ids, attention_mask = [el.to(device) for el in batch]
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, teacher_forcing=args["teacher_forcing"])

                batch_metrics = {
                    "model": out.loss,
                    "recon": out.recon_loss,
                    "kld": out.kld_loss,
                    "var": out.var_loss
                }
                
                vae_latent_repr = out.vae_latent_repr.view(-1, out.vae_latent_repr.shape[-1])
                fake = torch.randn_like(vae_latent_repr)

                vae_latent_repr_preds = discriminator(vae_latent_repr)
                fake_preds = discriminator(fake)

                ce_loss_fn = CrossEntropyLoss()
                batch_metrics["adv"] = ce_loss_fn(vae_latent_repr_preds, torch.ones(vae_latent_repr_preds.shape[0], dtype=torch.long).to(device))
                batch_metrics["d"] = ce_loss_fn(vae_latent_repr_preds.detach(), torch.zeros(vae_latent_repr_preds.shape[0], dtype=torch.long).to(device)) + ce_loss_fn(fake_preds, torch.ones(fake_preds.shape[0], dtype=torch.long).to(device))

                batch_metrics["total"] = sum([args[f"f_{metric}"]*batch_metrics[metric] for metric in _metrics[:-2]])

                for metric in _metrics:
                    metrics[f"validation/loss/{metric}/step"].append(batch_metrics[{metric}].item())
                    epoch_metrics[f"validation/{metric}/model"].append(batch_metrics[{metric}].item())

        for mode in _modes:
            for metric in _metrics:
                for reduction, f in {"mean": torch.mean, "median": torch.median}.items():
                    metrics[f"{mode}/loss/{metric}/{reduction}"].append(f(torch.tensor(epoch_metrics[f"{mode}/loss/{metric}"])))

        if args["earlystop_mode"] is None:
            stop = False
            _, save = earlystop(val_loss=metrics["validation/loss/total/mean"][-1], i_epoch=i_epoch)
        elif args["earlystop_mode"] == "mean":
            stop, save = earlystop(val_loss=metrics["validation/loss/total/mean"][-1], i_epoch=i_epoch)
        elif args["earlystop_mode"] == "median":
            stop, save = earlystop(val_loss=metrics["validation/loss/total/median"][-1], i_epoch=i_epoch)
        else:
            raise NotImplementedError()
        
        if save:
            model.save_pretrained(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "checkpoints", str(i_epoch)))
            model.save_pretrained(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best"))
            earlystop.save(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "checkpoints", str(i_epoch), "earlystop.json"))
            torch.save(optimizer.state_dict(), os.path.join(args["model_dir"], args["model_name"], args["model_version"], "checkpoints", str(i_epoch), "optimizer.bin"))

        torch.save(metrics, os.path.join(args["model_dir"], args["model_name"], args["model_version"], "metrics.bin"))

        str_metrics = f"Total Loss: [{metrics['train/loss/total/mean'][-1]:.4f};{metrics['train/loss/total/median'][-1]:.4f}] / [{metrics['validation/loss/total/mean'][-1]:.4f};{metrics['validation/loss/total/median'][-1]:.4f}] \n" + \
                      f"Model Loss: [{metrics['train/loss/model/mean'][-1]:.4f};{metrics['train/loss/model/median'][-1]:.4f}] / [{metrics['validation/loss/model/mean'][-1]:.4f};{metrics['validation/loss/model/median'][-1]:.4f}] \n" + \
                      f"Recon Loss: [{metrics['train/loss/recon/mean'][-1]:.4f};{metrics['train/loss/recon/median'][-1]:.4f}] / [{metrics['validation/loss/recon/mean'][-1]:.4f};{metrics['validation/loss/recon/median'][-1]:.4f}] \n" + \
                      f"KLD   Loss: [{metrics['train/loss/kld/mean'][-1]:.4f};{metrics['train/loss/kld/median'][-1]:.4f}] / [{metrics['validation/loss/kld/mean'][-1]:.4f};{metrics['validation/loss/kld/median'][-1]:.4f}] \n" + \
                      f"ADV   Loss: [{metrics['train/loss/adv/mean'][-1]:.4f};{metrics['train/loss/adv/median'][-1]:.4f}] / [{metrics['validation/loss/adv/mean'][-1]:.4f};{metrics['validation/loss/adv/median'][-1]:.4f}] \n" + \
                      f"Discr Loss: [{metrics['train/loss/d/mean'][-1]:.4f};{metrics['train/loss/d/median'][-1]:.4f}] / [{metrics['validation/loss/d/mean'][-1]:.4f};{metrics['validation/loss/d/median'][-1]:.4f}]"

        str_duration = f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
        str_earlystop = str(earlystop) if args["earlystop_mode"] is not None else ""
        str_epoch = f"EPOCH[{i_epoch}]" + "\n" + str_metrics + "\n" + str_earlystop + "\n" + str_duration
        logging.info(str_epoch)

    model.save_pretrained(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final"))

def main(
        args: typing.Dict
) -> None:

    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_config"])
    collate_fn = get_collate_fn(tokenizer)
    model = T5VAEForConditionalGeneration.from_pretrained(args["from_model"], dims_hidden=args["dims_hidden"], dim_latent=args["dim_latent"], p=args["p"])
    discriminator = nn.Sequential(
        nn.Linear(args["dim_latent"], args["dim_discriminator"]), 
        nn.ReLU(),
        nn.Linear(args["dim_discriminator"], 2), 
    )
    optimizer = AdamW(params=model.parameters(), lr=args["learning_rate"])
    d_optimizer = AdamW(params=discriminator.parameters(), lr=args["d_learning_rate"])

    training_dataset = Dataset(args["training_set"], args["min_length"], args["max_length"])
    validation_dataset = Dataset(args["validation_set"], args["min_length"], args["max_length"])

    training_dataloader = DataLoader(training_dataset, args["mini_batch_size"], True, collate_fn=collate_fn, num_workers=args["num_workers"])
    validation_dataloader = DataLoader(validation_dataset, args["mini_batch_size"], False, collate_fn=collate_fn, num_workers=args["num_workers"])
    logging.info(f"Dataset size:\n\t- training: {len(training_dataset)}\n\t- validation: {len(validation_dataset)}")

    earlystop = EarlyStop(args["patience"], args["warmup"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        train(tokenizer, model, discriminator, optimizer, d_optimizer, training_dataloader, validation_dataloader, collate_fn, earlystop, device, args)
    except KeyboardInterrupt:
        print("Exit training...")
        model.save_pretrained(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final"))


if __name__ == "__main__":
    args = parse_args()
    main(args)