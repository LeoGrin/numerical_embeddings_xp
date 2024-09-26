import torch
from torch import optim
from datasets import DatasetDict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import hashlib

# Where the model and collator is defined
from xval import numformer

### Huggingface dataset and tokenizer imports
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

# ### xVal imports
from xval import make_tokenizer, preprocess, analyze
### Define model
# The vocab_size is the number of different tokens in the tokenizer.
# context length is the maximum sequence size. 
import submitit
import argparse
import os
import fire
from datetime import datetime
import math
from torch.optim.lr_scheduler import LambdaLR

# copied from tabpfn (changed to have a minimum lr)
# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, min_lr=0.0):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * num_cycles * 2` after 
    a warmup period during which it increases linearly between min_lr and 1.
    """
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr + (1 - min_lr) * (float(current_step + 1) / float(max(1, num_warmup_steps)))
        else:
            progress = float(current_step + 1 - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            return min_lr + (1 - min_lr) * cos_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_xval(embedding_type="fourier",
               reverse_numerical_embedding=False,
               gaussian_embedding_sd=2.0,
               dim_feedforward=1536,
               dropout=0.1,
               norm_first=True,
               gaussian_embedding_left_bound=-5.5,
               gaussian_embedding_right_bound=5.5,
               gaussian_embedding_normalization="step",
               gaussian_embedding_embed_grid="equally_spaced",
               normalize_pos_enc=True,
               normalize_token_emb=True,
               parametrize_std=False,
               fourier_length_scale=1.0,
               fourier_kernel="gaussian",
               num_layers=3,
                nhead=3,
                d_model=384,
                context_length=955,
               iter=0,
               epochs=30,
               load_checkpoint=False):
    print("Training xVal model...")
    # params for planetary motion
    lr = 2e-5
    weight_decay = 0.1
    minimum_lr = 2e-6
    warmup_steps = 2000
    context_length = 955 #767 #CHANGED
    batch_size = 128
    d_model = 384#768
    dim_feedforward = 1536#3072
    vocab_size = 27
    nhead = 4#6
    num_layers = 6#3#6
    num_steps = 3_000_000
    mlm_probability = 0.2
    epochs = 200 # will be stopped before if num_steps is reached

    model = numformer.Numformer(vocab_size=vocab_size, nhead=nhead, num_layers=num_layers, d_model=d_model,  
                                dim_feedforward=dim_feedforward, context_length=context_length, numerical_embedding=embedding_type,
                                gaussian_embedding_sd=gaussian_embedding_sd, reverse_numerical_embedding=reverse_numerical_embedding,
                                parametrize_std=parametrize_std, gaussian_embedding_left_bound=gaussian_embedding_left_bound,
                                gaussian_embedding_right_bound=gaussian_embedding_right_bound, gaussian_embedding_embed_grid=gaussian_embedding_embed_grid,
                                fourier_length_scale=fourier_length_scale,
                                fourier_kernel=fourier_kernel, dropout=dropout, norm_first=norm_first,
                                gaussian_embedding_normalization=gaussian_embedding_normalization,
                                normalize_pos_enc=normalize_pos_enc, normalize_token_emb=normalize_token_emb).cuda()

    
    #TODO

    config = {
        "embedding_type": embedding_type,
        "reverse_numerical_embedding": reverse_numerical_embedding,
        "gaussian_embedding_sd": gaussian_embedding_sd,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "norm_first": norm_first,
        "gaussian_embedding_left_bound": gaussian_embedding_left_bound,
        "gaussian_embedding_right_bound": gaussian_embedding_right_bound,
        "gaussian_embedding_normalization": gaussian_embedding_normalization,
        "parametrize_std": parametrize_std,
        "normalize_pos_enc": normalize_pos_enc,
        "normalize_token_emb": normalize_token_emb,
        "fourier_length_scale": fourier_length_scale,
        "fourier_kernel": fourier_kernel,
        "num_layers": num_layers,
        "nhead": nhead,
        "d_model": d_model,
        "context_length": context_length,
        "iter": iter,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "minimum_lr": minimum_lr,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "vocab_size": vocab_size,
        "mlm_probability": mlm_probability,
        "dataset": "fad",
    }



    
    print(f"Using config: {config}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #we use a cosine learning-rate schedule with warm-up. The scheduler is
    # adjusted such that it reaches the minimum learning rate at the end of the training run.
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=1, eta_min=minimum_lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, num_cycles=0.5,
                                                min_lr=minimum_lr)
    

    ### Load the tokenizer 
    #tokenizer_path = "./tokenizer.json"
    #tokenizer_path = "./data/tokenized_fad_4tokenizer_xval.json"
    tokenizer_path = "./data/tokenized_fad_bigtokenizer_xval.json"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="[END]",
        eos_token="[END]",
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    pad_token_id = tokenizer.pad_token_id
    num_token_id = tokenizer.convert_tokens_to_ids("[NUM]")
    mask_token_id = tokenizer.mask_token_id
    #epochs = 10

    ### Load tokenized datasets 
    #dataset_path = "./data/tokenized_ds_xval"
    #dataset_path = "./data/tokenized_fad_big/tokenized_ds_xval"
    dataset_path = "./data/tokenized_fad_4_op_uniform/tokenized_ds_xval"
    #dataset_path = "./data/tokenized_fad_4_op_uniform_3/tokenized_ds_xval"
    #dataset_path = "./data/tokenized_fad_6_op/tokenized_ds_xval"
    tokenized_ds = DatasetDict.load_from_disk(dataset_path)

    # add dataset to config
    config["dataset_path"] = dataset_path
    config["tokenizer_path"] = tokenizer_path

    # Define the masked xVal collator which takes samples of unequal length and masks out both the token_ids and the numbers.
    #collator = numformer.define_masked_num_collator(pad_token_id, mask_token_id, mlm_probability)
    collator = numformer.define_masked_num_collator_last(pad_token_id, mask_token_id)
    #collator = numformer.define_masked_num_collator_one(pad_token_id, mask_token_id)

    config["collator"] = str(collator)

    hash = hashlib.sha256(str(config).encode()).hexdigest()

    train_loader = DataLoader(
        tokenized_ds["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        tokenized_ds["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    ### Run training loop

    loss_hist = []
    loss_mlm_hist = []
    loss_num_hist = []
    val_loss_hist = []
    val_loss_mlm_hist = []
    val_loss_num_hist = []

    # find date
    now = datetime.now()
    # Extract the day and the month
    day = now.day
    month = now.month

    # check if there is a checkpoint to load
    if load_checkpoint:
        checkpoints_dir = "./checkpoints"

        # Search for the checkpoint file with the specified hash
        checkpoint_file = None
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".pt") and hash in f:
                checkpoint_file = f
                break

        # Check if the checkpoint file was found
        if checkpoint_file:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found with hash {hash}.")

        if checkpoint_file:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            # Extract components
            model_state_dict = checkpoint['model']
            optimizer_state_dict = checkpoint['optimizer']
            loss_avg = checkpoint['loss']
            loss_hist = checkpoint['loss_hist']
            loss_mlm_hist = checkpoint['loss_mlm_hist']
            loss_num_hist = checkpoint['loss_num_hist']
            val_loss_hist = checkpoint['val_loss_hist']
            val_loss_mlm_hist = checkpoint['val_loss_mlm_hist']
            val_loss_num_hist = checkpoint['val_loss_num_hist']

            # Load state dictionaries into the model and optimizer if they exist
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            # find the number of steps for the scheduler
            n_epochs_checkpoint = len(loss_hist)
            n_batches_checkpoint = n_epochs_checkpoint * len(train_loader) 
            for _ in range(n_batches_checkpoint):
                scheduler.step()

            # Optionally, handle other data (like loss histories) according to your application needs
            print("Model and optimizer states have been restored.")
            print("Scheduler has been tentatively restored.")
            print("Epochs: ", n_epochs_checkpoint)
            print("Batches: ", n_batches_checkpoint)





    n_batches = 0


    try: 
        for e in tqdm(range(epochs)):
            loss_list_batch = []
            loss_mlm_list_batch = []
            loss_num_list_batch = []
            for batch in train_loader:
                logit_preds, num_preds = model(batch["x"].cuda(), batch["x_num"].cuda())
                with torch.autocast(device_type="cuda"):
                    loss_mlm = F.cross_entropy(
                        logit_preds.view(-1, logit_preds.size(-1)),
                        batch["y"].cuda().view(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )
                    num_mask = batch['y']==num_token_id
                    loss_num = F.mse_loss(
                        num_preds[num_mask],
                        batch["y_num"][num_mask].view(-1,1).cuda(),
                        reduction="mean",
                    )
                loss = loss_mlm + loss_num
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_list_batch.append(loss.item())
                loss_mlm_list_batch.append(loss_mlm.item())
                loss_num_list_batch.append(loss_num.item())
                n_batches += 1 
                # calculate the running average of the losses
                try:
                    loss_avg = 0.99*loss_avg + 0.01*loss.item()
                    loss_mlm_avg = 0.99*loss_mlm_avg + 0.01*loss_mlm.item()
                    loss_num_avg = 0.99*loss_num_avg + 0.01*loss_num.item()
                except:
                    loss_avg = loss.item()
                    loss_mlm_avg = loss_mlm.item()
                    loss_num_avg = loss_num.item()

                if n_batches % 1000 == 0:
                    print(f"Epoch #{e}: loss_mlm = {loss_mlm.item():.3f}; loss_num = {loss_num.item():.3f}; loss_total = {loss.item():.3f}")
                    print(f"Epoch #{e}: loss_mlm_avg = {loss_mlm_avg:.3f}; loss_num_avg = {loss_num_avg:.3f}; loss_avg = {loss_avg:.3f}")
            
            loss_hist.append(np.mean(loss_list_batch))
            loss_mlm_hist.append(np.mean(loss_mlm_list_batch))
            loss_num_hist.append(np.mean(loss_num_list_batch))

            
            ### Save checkpoint every 5 epochs
            if e % 1 == 0:
                # evaluate on the validation set
                model.eval()
                with torch.no_grad():
                    val_loss = []
                    val_loss_mlm = []
                    val_loss_num = []
                    for batch in val_loader:
                        logit_preds, num_preds = model(batch["x"].cuda(), batch["x_num"].cuda())
                        with torch.autocast(device_type="cuda"):
                            loss_mlm = F.cross_entropy(
                                logit_preds.view(-1, logit_preds.size(-1)),
                                batch["y"].cuda().view(-1),
                                ignore_index=-100,
                                reduction="mean",
                            )
                            num_mask = batch['y']==num_token_id
                            loss_num = F.mse_loss(
                                num_preds[num_mask],
                                batch["y_num"][num_mask].view(-1,1).cuda(),
                                reduction="mean",
                            )
                        loss = loss_mlm + loss_num
                        val_loss.append(loss.item())
                        val_loss_mlm.append(loss_mlm.item())
                        val_loss_num.append(loss_num.item())
                    val_loss_hist.append(np.mean(val_loss))
                    val_loss_mlm_hist.append(np.mean(val_loss_mlm))
                    val_loss_num_hist.append(np.mean(val_loss_num))
                model.train()
            if e % 3 == 0:
                checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "loss": loss_avg,
                        "loss_hist": loss_hist,
                        "loss_mlm_hist": loss_mlm_hist,
                        "loss_num_hist": loss_num_hist,
                        "val_loss_hist": val_loss_hist,
                        "val_loss_mlm_hist": val_loss_mlm_hist,
                        "val_loss_num_hist": val_loss_num_hist,
                    }
                torch.save(checkpoint, f"./checkpoints/xval_{day}_{month}_epoch_{e}_{hash}.pt")
            print(f"Epoch #{e}: loss_mlm = {loss_mlm_avg:.3f}; loss_num = {loss_num_avg:.3f}; loss_total = {loss_avg:.3f}")
            # save results in a csv file
            if config["embedding_type"] == "gaussian":
                std_dev = model.numerical_embedder.std_dev_list
            else:
                std_dev = None
            # convert config values to string to save
            for key, value in config.items():
                config[key] = str(value)
            results = pd.DataFrame({
                "epoch": list(range(len(loss_hist))),
                "loss_mlm": loss_mlm_hist,
                "loss_num": loss_num_hist,
                "loss_total": loss_hist,
                "val_loss_mlm": val_loss_mlm_hist,
                "val_loss_num": val_loss_num_hist,
                "val_loss_total": val_loss_hist,
                "std_dev": str(std_dev),
                **config
            })

            results.to_csv(f"./results/xval_fad_uniform_small_lengthscale_{day}_{month}_{hash}.csv", index=False)

            if n_batches > num_steps:
                print(f"Finished training after {e} epochs (n_batches = {n_batches})")
                break
            
    except KeyboardInterrupt:
        print('Interrupted')
        pass

if __name__ == "__main__":
    fire.Fire(train_xval)
