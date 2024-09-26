import torch
from torch import optim
from datasets import DatasetDict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import itertools

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
from train_xval_args import train_xval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="margaret")
    parser.add_argument("--n-gpus", type=int, default=6)
    args = parser.parse_args()

    base_length_scale = 2. / 384

    configs_dict = {
        "embedding_type": ["fourier"],
        "reverse_numerical_embedding": [False],
        # "fourier_kernel": ["gaussian", "laplacian", "cauchy"],
        # "fourier_length_scale": [0.5, 1.0, 2.0]
        "fourier_kernel": ["gaussian"],
        "fourier_length_scale": [base_length_scale / 3, base_length_scale, base_length_scale * 5, base_length_scale * 10, base_length_scale * 15],
        #"epochs": [100],
    }



    config_keys = list(configs_dict.keys())
    config_values = list(configs_dict.values())

    all_configs = list(itertools.product(*config_values))

    config_list = []
    for config in all_configs:
        config_dict = dict(zip(config_keys, config))
        config_list.append(config_dict)


    executor = submitit.AutoExecutor(folder="submitit_logs")
    array_parallelism = args.n_gpus
    if args.cluster == "margaret":
        print("On margaret, using submitit")
        executor.update_parameters(timeout_min=10000, slurm_partition='parietal,normal,gpu-best,gpu', slurm_array_parallelism=array_parallelism,#, cpus_per_task=2,
                                    gpus_per_node=1, exclude="margpu001,margpu002,margpu003,margpu004")

        with executor.batch():
            for config in config_list:
                for iter in range(1):
                    config_copy = config.copy()
                    config_copy.update({"iter": iter})
                    print(f"Submitting iter={iter}")
                    executor.submit(train_xval, **config_copy)
                    print("Submitted")
    elif args.cluster == "jz":
        print("On jz, creating sbatch files")
        for config in config_list:
            for iter in range(3):
                config_copy = config.copy()
                config_copy.update({"iter": iter})
                config_string = "_".join([f"{k}={v}" for k, v in config_copy.items()])
                # hash the config
                hash = hashlib.sha256(str(config_copy).encode()).hexdigest()
                with open(f"sbatch_files/sbatch_{hash}.sh", "w") as f:
                    f.write(f"#!/bin/bash\n")
                    f.write(f"#SBATCH --job-name=xval_{hash}\n")
                    f.write(f"#SBATCH --output=xval_{hash}.out\n")
                    f.write(f"#SBATCH --error=xval_{hash}.err\n")
                    f.write(f"#SBATCH -n 1\n")
                    f.write("#SBATCH --cpus-per-task=10\n")
                    f.write("#SBATCH --ntasks-per-node=1\n")
                    f.write("#SBATCH --gpus-per-task=1\n")
                    f.write("#SBATCH --hint=nomultithread\n")
                    f.write("#SBATCH --time=20:00:00\n")
                    f.write("#SBATCH -A ptq@v100\n")
                    f.write("#SBATCH -C v100-32g\n")
                    #f.write("#SBATCH --account=def-lgrinszt\n")

##SBATCH --ntasks-per-node=${NUM_GPUS_PER_NODE}
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:${NUM_GPUS_PER_NODE}
#SBATCH --gpus-per-task=${NUM_GPUS_PER_NODE}
#SBATCH --hint=nomultithread         # hyperthreading desactive
                    f.write(f"module load pytorch-gpu/py3/2.1.1\n")
                    #f.write(f"conda activate numerical_embeddings\n")
                    command = "python train_xval_args.py"
                    for key, value in config_copy.items():
                        command += f" --{key} {value}"
                    #f.write(f"python train_xval_args.py --{param} {variant} --iter {iter}\n")
                    f.write(command)
                # submit the sbatch file
                print(f"Submitting iter={iter}")
                os.system(f"sbatch sbatch_files/sbatch_{hash}.sh")
                print("Submitted")
    else:
        print("Unknown cluster")
