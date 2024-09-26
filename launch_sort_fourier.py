from model import TransformerModel, TransformerModelWithEmbeddings
from tasks import RandomSequenceIdentityDataset, RandomSequenceSortDataset, RandomSequencePolynomialDataset, RandomSequenceMedianRankDataset, RandomSequenceSumDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
from utils import deactivate_positional_encoding, deactivate_layer_norm
from embeddings import PositionalEncoding, GaussianEmbedder, RandomFeatureEmbedder
import pickle
from train_sort import train

import submitit
config = {
    "model": {
        "in_seq_length": 10,
        "out_seq_length": 10,
        "n_layer": 8,
        "n_head": 4,
        "n_embd": 128,
    },
    "n_epochs": 300,
    "n_samples": 10_000,
    "seq_length": 10,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 0,
    "task": "sort",
    "verbose": 0,
    "no_positional_encoding": False,
    "no_layer_norm": False,
    "fixed_embeddings": False,
    "classif": False,
    "device": "cpu",#"cuda:0",
    "save": True,
}

# launching for GaussianEmbedder

executor = submitit.AutoExecutor(folder="submitit_logs")
array_parallelism = 110
#executor.update_parameters(timeout_min=2000, slurm_partition='parietal,normal,gpu-best,gpu', slurm_array_parallelism=array_parallelism,#, cpus_per_task=2,
#                             gpus_per_node=1)
                           # exclude="margpu009")
executor.update_parameters(timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=array_parallelism, cpus_per_task=2,
                          exclude="margpu009")

# embedding_function_list = ["gaussian", None]
# read_in_bias_list = [True, False]
# only_one_emb_list = [True, False]
lr_list = [1e-4]
depth_list = [8, 4, 2, 1]
n_head_list = [4]
width_list = [128, 256]
attention_only_list = [True]
seq_length_list = [10]
length_scale_list = [(12 / 128) * 5]
#kernel_list = ["gaussian", "laplacian", "cauchy"]
kernel_list = ["gaussian"]
#holes_list = [[(3, 7)], [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]]
holes_list = [[]]


n_iter = 5
# compute the number of jobs

config_list = []

for lr in lr_list:
    for depth in depth_list:
        for n_head in n_head_list:
            for width in width_list:
                for attention_only in attention_only_list:
                        for seq_length in seq_length_list:
                            for length_scale in length_scale_list:
                                for kernel in kernel_list:
                                    for holes in holes_list:
                                        for iter in range(n_iter):
                                            config_copy = config.copy()
                                            config_copy["model"] = config["model"].copy()
                                            config_copy["model"].update({
                                                "n_layer": depth,
                                                "n_head": n_head,
                                                "n_embd": width,
                                                "attention_only": attention_only,
                                                "revert_embedding": False,
                                                "embedding_function": RandomFeatureEmbedder(num_random_features=width,
                                                                                            kernel=kernel,
                                                                                            length_scale=length_scale,
                                                ),
                                                "in_seq_length": seq_length,
                                                "out_seq_length": seq_length,
                                            })
                                            config_copy.update({
                                                "lr": lr,
                                                "iter": iter,
                                                "seq_length": seq_length,
                                                "length_scale": length_scale,
                                                "kernel": kernel,
                                                "holes": holes,
                                            })
                                            config_list.append(config_copy)

print("Number of jobs", len(config_list))
                            

max_job_per_array = 300
# create chunks of the config list

for i in range(0, len(config_list), max_job_per_array):
    config_list_chunk = config_list[i:i+max_job_per_array]
    # add the job_id to the config
    for j, config in enumerate(config_list_chunk):
        config["job_id"] = "sort_ff_13_05"
    print("Submitting", len(config_list_chunk), "jobs")
    executor.map_array(train, config_list_chunk)
    print("Submitted")
    #for config in config_list_chunk:
    #    train(config)