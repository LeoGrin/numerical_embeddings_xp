# Code for the Paper Encoding numerical values for Transformers


## Installation

Required packages can be installed via `pip install -r requirements.txt`


## Getting the data:

- Arithmetic experiment: the data can be generated using the `arithmetic_data_set.ipynb` notebook

- Planetary orbits: the data can be generated following the instructions in `xval/planet_sims`

- Sorting: the data are generated on the fly during training.

## Running the experiments:

For all the experiement files, some cluster settings (partition...) will need to be adapted to your situation.

- Sorting: `python launch_sort_{embedding}.py`

- Planetary orbits and Arithmetic: `python launch_xval_{embedding}.py`, after setting the dataset and tokenizer path to the correct dataset in `train_xval_args.py`

- Toy KNN experiment: available in `differentiable_knn.ipynb`.


## Credit

Code in the xval folder is adapted from the code of the original xval paper, available at: https://github.com/PolymathicAI/xVal

