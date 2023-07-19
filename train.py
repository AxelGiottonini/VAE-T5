import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from src import configure, adversarial_train_loop, get_model, get_dataloaders


random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    args = configure()
    training_dataloader, validation_dataloader = get_dataloaders(args)
    model, discriminator, optimizer, d_optimizer = get_model(args, return_discriminator=True, return_optimizer=True)

    ce_loss_fn = CrossEntropyLoss()

    @adversarial_train_loop(
        model=model,
        discriminator=discriminator,
        optimizer=optimizer,
        d_optimizer=d_optimizer,
        args=args
    )
    def train(model, discriminator, batch):
        input_ids = batch.masked_input_ids
        attention_mask = batch.attention_mask
        output_ids = batch.input_ids

        print(input_ids.shape, attention_mask.shape, output_ids.shape)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids, teacher_forcing=args["teacher_forcing"])

        vae_latent_repr = out.vae_latent_repr.view(-1, out.vae_latent_repr.shape[-1])
        fake_latent_repr = torch.tensor(
            np.random.multivariate_normal(
                np.mean(vae_latent_repr.float().cpu().detach().numpy(), axis=0),
                np.cov(vae_latent_repr.float().cpu().detach().numpy()),
            )
        ).to(vae_latent_repr.device).to(vae_latent_repr.dtype)

        vae_latent_repr_cls = discriminator(vae_latent_repr)
        fake_latent_repr_cls = discriminator(fake_latent_repr)

        adv_loss = ce_loss_fn(vae_latent_repr_cls, torch.ones(vae_latent_repr_cls.shape[0], dtype=torch.long).to(vae_latent_repr_cls.device))
        d_loss = ce_loss_fn(vae_latent_repr_cls.detach(), torch.zeros(vae_latent_repr_cls.shape[0], dtype=torch.long).to(vae_latent_repr_cls.device)) + ce_loss_fn(fake_latent_repr_cls, torch.ones(fake_latent_repr_cls.shape[0], dtype=torch.long).to(fake_latent_repr_cls.device))

        loss = args["f_model"] * out.loss + \
               args["f_recon"] * out.recon_loss + \
               args["f_kld"] * out.kld_loss + \
               args["f_var"] * out.var_loss + \
               args["f_adv"] * adv_loss 
        
        return loss, d_loss

    train(training_dataloader, validation_dataloader)