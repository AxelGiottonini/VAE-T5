from typing import Optional, Tuple, Union

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class Seq2SeqVAELMOutput(Seq2SeqLMOutput):

    recon_loss: Optional[torch.FloatTensor] = None
    kld_loss: Optional[torch.FloatTensor] = None
    var_loss: Optional[torch.FloatTensor] = None
    vae_latent_repr: Optional[Tuple[torch.FloatTensor]] = None
    vae_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class T5VAEForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, dims_hidden, dim_latent, p):
        super().__init__(config)

        encoder = []
        encoder_input_layer = nn.Linear(config.d_model, dims_hidden[0])
        for dim_in, dim_out in zip(dims_hidden[:-1], dims_hidden[1:]):
            encoder.append(
                nn.Sequential(
                    nn.Dropout(p),
                    nn.Linear(dim_in, dim_out),
                    nn.ReLU()
                )
            )
        self.vae_encoder = nn.Sequential(
            encoder_input_layer,
            *encoder
        )

        self.vae_fc_mu = nn.Linear(dims_hidden[-1], dim_latent)
        self.vae_fc_var = nn.Linear(dims_hidden[-1], dim_latent)

        dims_hidden_reversed = dims_hidden[::-1]
        decoder = []
        decoder_input_layer = nn.Linear(dim_latent, dims_hidden_reversed[0])
        decoder_output_layer = nn.Sequential(
            nn.Linear(dims_hidden_reversed[-1], config.d_model),
            nn.Tanh()
        )
        for dim_in, dim_out in zip(dims_hidden_reversed[:-1], dims_hidden_reversed[1:]):
            decoder.append(
                nn.Sequential(
                    nn.Dropout(p),
                    nn.Linear(dim_in, dim_out),
                    nn.ReLU()
                )
            )
        self.vae_decoder = nn.Sequential(
            decoder_input_layer,
            *decoder,
            decoder_output_layer
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_forcing: Optional[float] = None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        encoder_outputs = self.encode(
            input_ids, 
            attention_mask, 
            head_mask, 
            encoder_outputs, 
            inputs_embeds, 
            output_attentions, 
            output_hidden_states, 
            return_dict
        )

        hidden_states = encoder_outputs[0]
        recon, vae_z, recon_loss, kld_loss, var_loss = self.autoencode(hidden_states, attention_mask)

        if teacher_forcing is None:
            hidden_states = recon
        else:
            coefficients = torch.rand_like(hidden_states)
            hidden_states = (coefficients <= teacher_forcing).to(torch.bfloat16)*hidden_states + (coefficients > teacher_forcing).to(torch.bfloat16)*recon

        decoder_outputs = self.decode(
            hidden_states,
            attention_mask, 
            decoder_input_ids, 
            decoder_attention_mask, 
            decoder_head_mask, 
            cross_attn_head_mask,
            encoder_outputs, 
            past_key_values, 
            decoder_inputs_embeds, 
            labels, 
            use_cache, 
            output_attentions,
            output_hidden_states, 
            return_dict
        )
        
        decoder_outputs.recon_loss=recon_loss
        decoder_outputs.kld_loss=kld_loss
        decoder_outputs.var_loss=var_loss
        decoder_outputs.vae_latent_repr=vae_z
        decoder_outputs.vae_hidden_states=recon

        return decoder_outputs

    def autoencode(
        self,
        hidden_states,
        attention_mask
    ):
        vae_h = self.vae_encoder(hidden_states)
        vae_mu = self.vae_fc_mu(vae_h)
        vae_logvar = self.vae_fc_var(vae_h)
        vae_std = torch.exp(0.5*vae_logvar)
        vae_eps = torch.randn_like(vae_std)
        vae_z = vae_eps * vae_std + vae_mu
        recon = self.vae_decoder(vae_z)

        recon_loss_fn = MSELoss(reduction='sum')
        recon_loss = recon_loss_fn(
            attention_mask.view(-1, 1) * recon.view(-1, recon.shape[-1]), 
            attention_mask.view(-1, 1) * hidden_states.view(-1, hidden_states.shape[-1])
        )
        recon_loss = recon_loss / attention_mask.sum()

        kld_loss = -0.5 * torch.sum(1 + vae_logvar.view(-1, vae_logvar.shape[-1]) - vae_mu.view(-1, vae_mu.shape[-1]) ** 2 - vae_logvar.view(-1, vae_logvar.shape[-1]).exp(), dim = 1)
        kld_loss = torch.sum(attention_mask.view(-1) * kld_loss) / attention_mask.sum()

        var_loss = (attention_mask.view(-1, 1) * vae_logvar.view(-1, vae_logvar.shape[-1])).abs().sum(dim=-1)
        var_loss = torch.sum(var_loss) / attention_mask.sum()

        return recon, vae_z, recon_loss, kld_loss, var_loss

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return encoder_outputs

    def decode(
        self,
        hidden_states, 
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqVAELMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )