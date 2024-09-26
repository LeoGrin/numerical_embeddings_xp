from model import TransformerModel, TransformerModelWithEmbeddings
from tasks import RandomSequenceIdentityDataset, RandomSequenceSortDataset, RandomSequencePolynomialDataset,\
      RandomSequenceMedianRankDataset, RandomSequenceSumDataset, RandomSequenceSortDatasetWithHoles, RandomSequenceKernelDataset, \
      RandomSequenceRankClosest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
from utils import deactivate_positional_encoding, deactivate_layer_norm
from embeddings import PositionalEncoding, GaussianEmbedder
import pickle
import hashlib

def train(config):
    
    config_model = config["model"]
    model = TransformerModelWithEmbeddings(**config_model)
    print(model)

    if config["no_positional_encoding"]:
        deactivate_positional_encoding(model)
    if config["no_layer_norm"]:
        deactivate_layer_norm(model)
    if config["fixed_embeddings"]:
        model._read_in.requires_grad = False
    
    # count the number of parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params}")
    # optimize both the embedding and the transformer encoder
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["classif"]:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    device = config["device"]
    model.to(device)

    # create dataloader
    if config["task"] == "identity":
        train_dataset = RandomSequenceIdentityDataset(config["n_samples"], config["seq_length"], config["seq_length"])
        test_dataset = RandomSequenceIdentityDataset(config["n_samples"], config["seq_length"], config["seq_length"])
    elif config["task"] == "sort":
        train_dataset = RandomSequenceSortDataset(config["n_samples"], config["seq_length"], config["seq_length"])
        test_dataset = RandomSequenceSortDataset(config["n_samples"], config["seq_length"], config["seq_length"])
    elif config["task"] == "sort_with_holes":
        train_dataset = RandomSequenceSortDatasetWithHoles(config["n_samples"], config["seq_length"], config["seq_length"],
                                                           holes=config["holes"])
        test_dataset = RandomSequenceSortDatasetWithHoles(config["n_samples"], config["seq_length"], config["seq_length"],
                                                            holes=[]) # no holes in the test set
    elif config["task"] == "polynomial":
        train_dataset = RandomSequencePolynomialDataset(config["n_samples"], config["seq_length"])
        test_dataset = RandomSequencePolynomialDataset(config["n_samples"], config["seq_length"])
    elif config["task"] == "median_rank":
        train_dataset = RandomSequenceMedianRankDataset(config["n_samples"], config["seq_length"])
        test_dataset = RandomSequenceMedianRankDataset(config["n_samples"], config["seq_length"])
    elif config["task"] == "sum":
        train_dataset = RandomSequenceSumDataset(config["n_samples"], config["seq_length"])
        test_dataset = RandomSequenceSumDataset(config["n_samples"], config["seq_length"])
    elif config["task"] == "kernel":
        train_dataset = RandomSequenceKernelDataset(config["n_samples"], config["seq_length"], kernel=config["kernel"],
                                                    length_scale=config["length_scale"])
        test_dataset = RandomSequenceKernelDataset(config["n_samples"], config["seq_length"], kernel=config["kernel"],
                                                    length_scale=config["length_scale"])
    elif config["task"] == "rank_closest":
        train_dataset = RandomSequenceRankClosest(config["n_samples"], config["seq_length"])
        test_dataset = RandomSequenceRankClosest(config["n_samples"], config["seq_length"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)


    hash = hashlib.sha256(str(config).encode()).hexdigest()
    loss_list = []
    test_loss_list = [] 

    for epoch in tqdm(range(config["n_epochs"])):
        try:
            for batch_x, batch_y in train_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if config["classif"]:
                    batch_y = batch_y.squeeze().long()
                start = time.time()
                out = model(batch_x)#[:, 0, :]
                #out = model(batch_x)[:, 0, :]
                #print("Forward pass took", time.time() - start, "seconds")
                if config["verbose"] > 0:
                    print("out", out.shape)
                    print("batch_y", batch_y.shape)
                loss = loss_fn(out, batch_y)
                optimizer.zero_grad()
                start = time.time()
                loss.backward()
                #print("Backward pass took", time.time() - start, "seconds")
                optimizer.step()
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, loss {loss.item()}")
                loss_list.append(loss.item())
            if epoch % 10 == 0:
                # show some examples
                for i in range(5):
                    x = batch_x[i, :, :].squeeze().detach().cpu().numpy()
                    #y = batch_y[i, :, :].squeeze().detach().cpu().numpy()
                    y = batch_y[i].squeeze().detach().cpu().numpy()
                    #y_pred = out[i, :, :].squeeze().detach().cpu().numpy()
                    y_pred = out[i, :].squeeze().detach().cpu().numpy()
                    # softmax
                    if config["classif"]:
                        y_pred_softmax = F.softmax(out[i, :], dim=-1)
                        # highest probability
                        best_guess = torch.argsort(out[i, :].squeeze())[-1]
                    print("x", x)
                    print("y", y)
                    print("y_pred", y_pred)
                    if config["classif"]:
                        print("y_pred_softmax", y_pred_softmax)
                        print("best_guess", best_guess)
            if epoch % 1 == 0 and test_dataloader is not None:
                # evaluate the model on the test set
                test_loss = 0
                n_test = 0
                for batch_x, batch_y in test_dataloader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    if config["classif"]:
                        batch_y = batch_y.squeeze().long()
                    out = model(batch_x)
                    loss = loss_fn(out, batch_y)
                    test_loss += loss.item()
                    n_test += 1
                test_loss /= n_test
                print(f"Test loss: {test_loss}")
                test_loss_list.append(test_loss)
                # save checkpoint
                if epoch % 50 == 0:
                    save_to = f"./checkpoints/{config['task']}_"
                    #torch.save(model.state_dict(), f"{save_to}_{epoch}_{hash}.pt")
                    # save model, and config and loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_list,
                        'test_loss': test_loss_list,
                        'config': config,
                        }, f"{save_to}_{epoch}_{hash}.pt")
        except KeyboardInterrupt:
            print("Interrupted")
            if config.get("smooth_interrupt", False):
                break
            raise KeyboardInterrupt
    #TODO: keep a simpler interactive function mode
    # save the results
    if "save" in config and config["save"]:
        if "job_id" not in config:
            # hash the config
            prefix = ""
        else:
            prefix = config["job_id"] + "_"
        job_id = prefix + hashlib.md5(str(config).encode()).hexdigest()
        with open(f"results/results_sort_{hash}.pkl", "wb") as f:
            pickle.dump((loss_list, test_loss_list, config), f)
    if config.get("return_model", False):
        return loss_list, test_loss_list, config, model
    return loss_list, test_loss_list, config