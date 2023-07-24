import typing
import torch
from dataclasses import dataclass

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

@dataclass
class VAEOutput():
    mu: typing.Optional[torch.Tensor]=None
    logvar: typing.Optional[torch.Tensor]=None
    latent: typing.Optional[torch.Tensor]=None
    recon: typing.Optional[torch.Tensor]=None
    loss: typing.Optional[torch.Tensor]=None

@dataclass
class Seq2SeqVAELMOutput(Seq2SeqLMOutput, VAEOutput):
    pass