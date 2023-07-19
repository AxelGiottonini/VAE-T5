import sys
import os
import time
import typing
import logging
import json
import torch

from .cli import summary

def adversarial_train_loop(
    model,
    discriminator,
    optimizer,
    d_optimizer,
    args,
    device: torch.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
    precision: torch.memory_format=torch.bfloat16,
    save_model: bool=True,
    save_optimizer: bool=True
):
    n_epochs = args["n_epochs"]
    global_batch_size = args["global_batch_size"]
    local_batch_size = args["local_batch_size"] if args["local_batch_size"] is not None else global_batch_size
    accumulation_steps = global_batch_size // local_batch_size
    save_each = args["save_each"]

    model.to(device).to(precision)
    discriminator.to(device).to(precision)

    metrics = {
        "best_loss": float("inf"),
        "training/loss/step": [],
        "training/loss/mean": [],
        "validation/loss/mean": []
    }

    def save(dir_name):
        if save_model:
            try:
                model.save(os.path.join(args["model_dir"], args["model_name"], args["model_version"], dir_name, "model.bin"))
            except AttributeError:
                torch.save(
                    model.state_dict(),
                    os.path.join(args["model_dir"], args["model_name"], args["model_version"], dir_name, "model.bin")
                )

        if save_optimizer:
            torch.save(
                optimizer.state_dict(), 
                os.path.join(args["model_dir"], args["model_name"], args["model_version"], dir_name, "optimizer.bin")
            )

    def decorator(step):

        def wrapper(training_dataloader, validation_dataloader=None):
            summary(model, training_dataloader, validation_dataloader)
            for i_epoch in range(1, n_epochs+1):
                try:
                    epoch_metrics = {
                        "start_time": time.time(),
                        "training_loss": [],
                        "validation/loss": []
                    }

                    model.train()
                    torch.cuda.empty_cache()
                    for i_batch, batch in enumerate(training_dataloader):
                        loss, d_loss = step(model, discriminator, batch.to(device))

                        (loss / accumulation_steps).backward()
                        (d_loss / accumulation_steps).backward()

                        if (
                            (i_batch + 1) % accumulation_steps == 0 or 
                            (i_batch + 1) == len(training_dataloader)
                        ):
                            optimizer.step()
                            optimizer.zero_grad()
                            d_optimizer.step()
                            d_optimizer.zero_grad()

                        metrics["training/loss/step"].append(loss.item())
                        epoch_metrics["training/loss"].append(loss.item())
                    
                        if (
                            (i_batch + 1) % (save_each * accumulation_steps) == 0 or
                            (i_batch + 1) == len(training_dataloader)
                        ):
                            torch.save(
                                metrics, 
                                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "metrics.bin")
                            )

                    metrics["training/loss/mean"].append(torch.tensor(epoch_metrics["training/loss"]).mean())

                    if validation_dataloader is not None:
                        model.eval()
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            for i_batch, batch in enumerate(validation_dataloader):
                                loss, _ = step(model, batch.to(device))
                                epoch_metrics["validation/loss"].append(loss.item())

                            metrics["validation/loss/mean"].append(torch.tensor(epoch_metrics["validation/loss"]).mean())

                    # Logging
                    str_metrics = f"Training Loss: {metrics['training/loss/mean'][-1]:.4f}"
                    if validation_dataloader is not None:
                        str_metrics = str_metrics + " | " + f"Validation Loss:{metrics['validation/loss/mean'][-1]:.4f}"
                    
                    str_duration = f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_metrics['start_time']))}"
                    str_epoch = f"EPOCH[{i_epoch}]" + "\n\t" + str_metrics + "\n\t" + str_duration
            
                    logging.info(str_epoch)

                    # Saving
                    if validation_dataloader is not None:
                        if metrics['validation/loss/mean'][-1] < metrics["best_loss"]*0.98:
                            save("best")
                            metrics["best_loss"] = metrics['validation/loss/mean'][-1]

                except (KeyboardInterrupt):
                    save("crash")
                    sys.exit(0)

            save("final")

        return wrapper    
    return decorator
