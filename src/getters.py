import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from .tokenizer import __get_collate_fn__
from .dataset import Dataset
from .modeling_t5vae import T5VAEForConditionalGeneration

def __get_tokenizer(args):
    return AutoTokenizer.from_pretrained(args["from_tokenizer"])

def __get_collate_fn(args, tokenizer):
    return __get_collate_fn__(tokenizer, mask=args["mask"], mask_rate=args["mask_rate"], frag_coef_a=args["frag_coef_a"], frag_coef_b=args["frag_coef_b"], split=args["split"])

def get_model(args, return_discriminator=False, return_optimizer=False):
    model = T5VAEForConditionalGeneration.from_pretrained(
        args["from_model"],
        dims_hidden=args["vae_dims_hidden"],
        dim_latent=args["vae_dim_latent"],
        p=args["vae_dropout_p"]
    )
    if return_optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["learning_rate"], 
            betas=args["betas"], 
            eps=args["eps"], 
            weight_decay=args["weight_decay"]
        )

    if return_discriminator:
        discriminator = nn.Sequential(
            nn.Linear(args["vae_dim_latent"], args["vae_dim_discriminator"]), 
            nn.ReLU(),
            nn.Linear(args["vae_dim_discriminator"], 2), 
        )
        if return_optimizer:
            d_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args["d_learning_rate"], 
                betas=args["d_betas"], 
                eps=args["d_eps"], 
                weight_decay=args["d_weight_decay"]
            )

    out = (
        model,
        discriminator if return_discriminator else None,
        optimizer if return_optimizer else None,
        d_optimizer if return_discriminator and return_optimizer else None
    )
    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out

def get_dataloaders(args, shuffle=True, return_validation=True, return_tokenizer=False, return_collate_fn=False):
    tokenizer = __get_tokenizer(args)
    collate_fn = __get_collate_fn(args, tokenizer)

    training_set = Dataset(args["training_set"], args["min_length"], args["max_length"])
    training_dataloader = DataLoader(dataset=training_set, batch_size=args["local_batch_size"], shuffle=shuffle, num_workers=args["num_workers"], collate_fn=collate_fn)
    
    if return_validation:
        validation_set = Dataset(args["validation_set"], args["min_length"], args["max_length"])
        validation_dataloader = DataLoader(dataset=validation_set, batch_size=args["local_batch_size"], shuffle=False, num_workers=args["num_workers"], collate_fn=collate_fn)

    out = (
        training_dataloader,
        validation_dataloader if return_validation else None,
        tokenizer if return_tokenizer else None,
        collate_fn if return_collate_fn else None
    )
    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out