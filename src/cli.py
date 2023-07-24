import os
import argparse
import logging
import json

def __parse_args__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_version", type=str, required=True, help="Movel version")

    parser.add_argument("--from_tokenizer", type=str, default="ElnaggarLab/ankh-base", help="Path or Huggingface's repository of the model's tokenizer")
    parser.add_argument("--from_model", type=str, default="ElnaggarLab/ankh-base", help="Path to repository containing the model's encoder and decoder")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--betas", type=str, default="(0.9, 0.999)", help="betas")
    parser.add_argument("--eps", type=float, default=1e-08, help="eps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")

    parser.add_argument("--d_learning_rate", type=float, default=0.001, help="Discriminator Optimizer Learning rate")
    parser.add_argument("--d_betas", type=str, default="(0.9, 0.999)", help="Discriminator Optimizer betas")
    parser.add_argument("--d_eps", type=float, default=1e-08, help="Discriminator Optimizer eps")
    parser.add_argument("--d_weight_decay", type=float, default=0.01, help="Discriminator Optimizer weight decay")

    parser.add_argument("--training_set", type=str, required=True, help="Path to training set")
    parser.add_argument("--validation_set", type=str, default=None, help="Path to validation set")
    parser.add_argument("--min_length", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length")

    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--mask_rate", type=float, default=0.15, help="masking probability")
    parser.add_argument("--frag_coef_a", type=float, default=0, help="Fragmentation coefficient A")
    parser.add_argument("--frag_coef_b", type=float, default=1, help="Fragmentation coefficient B")
    parser.add_argument("--split", action="store_true")

    parser.add_argument("--n_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--global_batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--local_batch_size", type=int, default=1, help="Mini-Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of sub-processes to use for data loading.")

    parser.add_argument("--vae_dims_hidden", type=str)
    parser.add_argument("--vae_dim_latent", type=int)
    parser.add_argument("--vae_dropout_p", type=float)
    parser.add_argument("--vae_dim_discriminator", type=int)

    parser.add_argument("--mode", type=str, default="standard")
    parser.add_argument("--token_wise", action="store_true")
    parser.add_argument("--f_recon", type=float, default=1)
    parser.add_argument("--f_kl", type=float, default=1)
    parser.add_argument("--f_adv", type=float, default=1)
    parser.add_argument("--f_logvar", type=float, default=1)
    parser.add_argument("--teacher_forcing", type=float, default=None)

    parser.add_argument("--save_each", type=int, default=10)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--precision", type=str, default="bfloat16")

    args = vars(parser.parse_args())

    betas = args["betas"][1:-1].replace(" ", "").split(",")
    if not len(betas) == 2:
        raise ValueError()
    args["betas"] = tuple(float(el) for el in betas)

    betas = args["d_betas"][1:-1].replace(" ", "").split(",")
    if not len(betas) == 2:
        raise ValueError()
    args["d_betas"] = tuple(float(el) for el in betas)

    args["vae_dims_hidden"] = [int(el) for el in args["vae_dims_hidden"].split(",")]

    if not args["global_batch_size"] % args["local_batch_size"] == 0:
        raise ValueError(f"--global_batch_size ({args['global_batch_size']}) should be a multiple of --local_batch_size ({args['local_batch_size']})")

    return args

def __safe_makedirs__(model_dir, model_name, model_version):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if not os.path.isdir(current:=(os.path.join(model_dir, model_name))):
        os.mkdir(current)

    if not os.path.isdir(current:=(os.path.join(model_dir, model_name, model_version))):
        os.mkdir(current)
        os.mkdir(os.path.join(current, "best"))
        os.mkdir(os.path.join(current, "final"))
        os.mkdir(os.path.join(current, "crash"))
    else:
        raise FileExistsError("The same version of the model exists, please choose a new version")

def __safe_logging__(log_dir, model_name, model_version):
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if not os.path.isdir(current:=(os.path.join(log_dir, model_name))):
        os.mkdir(current)

    logging.basicConfig(filename=os.path.join(log_dir, model_name, model_version + ".log"), level=logging.INFO, format='%(message)s')

def __save_args__(model_dir, model_name, model_version, args):
    with open(os.path.join(model_dir, model_name, model_version, "args.json"), 'w', encoding='utf8') as f:
        f.write(json.dumps(args, indent=4, sort_keys=False, separators=(',', ': '), ensure_ascii=False))

def configure():
    args = __parse_args__()
    __safe_makedirs__(args["model_dir"], args["model_name"], args["model_version"])
    __safe_logging__(args["log_dir"], args["model_name"], args["model_version"])
    __save_args__(args["model_dir"], args["model_name"], args["model_version"], args)
    return args

def summary(model, training_dataloader, validation_dataloader):
    n_total_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    len_training_dataloader = len(training_dataloader.dataset) if training_dataloader is not None else 0
    len_validation_dataloader = len(validation_dataloader.dataset) if validation_dataloader is not None else 0
    logging.info(
        f"\n"
        f"Parameters:\n" +
        f"\tTotal: {n_total_params}\n" +
        f"\tTraining: {n_train_params}\n" +
        f"="*80 + "\n" +
        f"Datasets:\n" +
        f"\tTraining: {len_training_dataloader}\n" +
        f"\tValidation: {len_validation_dataloader}\n" +
        f"="*80 + "\n"
    )