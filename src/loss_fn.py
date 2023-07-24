import torch
import torch.nn.functional as F

from .outputs import VAEOutput

def __recon_loss(output:VAEOutput):
    return F.mse_loss(output.recon.view(-1, output.recon.shape[-1]), output.encoder_last_hidden_state.view(-1, output.encoder_last_hidden_state.shape[-1]), reduction="mean")

def __kl_loss(output:VAEOutput):
    return -0.5 * torch.sum(1 + output.logvar - output.mu.pow(2) - output.logvar.exp()) / len(output.mu)

def __logvar_loss(output:VAEOutput):
    return output.logvar.abs().sum(dim=1).mean()

def __adv_loss(output:VAEOutput, discriminator, token_wise=False):
    fake = torch.randn_like(output.latent)

    if token_wise:
        zeros = torch.empty((output.latent.shape[0], output.latent.shape[1], 2))
        zeros[:, :, 0] = 1
        zeros[:, :, 1] = 0
        zeros = zeros.to(output.latent.device)
        ones = 1 - zeros
    else:
        zeros = torch.empty(len(output.latent), 2)
        zeros[:, 0] = 1
        zeros[:, 1] = 0
        zeros = zeros.to(output.latent.device)
        ones = 1 - zeros

    loss = F.binary_cross_entropy_with_logits(discriminator(output.latent), zeros)
    d_loss = F.binary_cross_entropy_with_logits(discriminator(output.latent.detach()), ones)
    d_loss = d_loss + F.binary_cross_entropy_with_logits(discriminator(fake), zeros)

    return loss, d_loss

def loss_fn(output, discriminator, args):
    if args["mode"] == "none":
        return output.loss

    if args["mode"] == "standard":
        return output.loss + \
            (args["f_recon"] * __recon_loss(output) if args["token_wise"] else 0)
    
    if args["mode"] == "vae":
        return output.loss + \
            (args["f_recon"] + __recon_loss(output) if args["token_wise"] else 0) + \
            args["f_kl"] * __kl_loss(output)
    
    if args["mode"] == "aae":
        adv_loss, d_loss = __adv_loss(output, discriminator, args["token_wise"])
        return (
            output.loss + \
                (args["f_recon"] * __recon_loss(output) if args["token_wise"] else 0) + \
                args["f_logvar"] * __logvar_loss(output) + \
                args["f_adv"] * adv_loss,
            d_loss
        )
    
    raise NotImplementedError()