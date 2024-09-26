from model import TransformerModel
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

checkpoint_dir = "checkpoints"


class RandomSequenceIdentityDataset(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=2, out_seq_length=1, range_left=-1, range_right=11):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.range_left = range_left
        self.range_right = range_right
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left
        x = x.unsqueeze(1)
        y = x[:self.out_seq_length]
        return x, y
    
# class RandomSequenceIdentityDatasetTest(Dataset):
#     def __len__(self):
#         return 10_000

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
#         x = x.unsqueeze(1)
#         y = x[0]
#         return x, y

class RandomSequenceSortDataset(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=2, out_seq_length=1, ascending=True, range_left=-1, range_right=11):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.ascending = ascending
        self.range_left = range_left
        self.range_right = range_right
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left 
        x = x.unsqueeze(1)
        y = torch.sort(x, dim=0)[0]
        if not self.ascending:
            y = y.flip(0)
        return x, y[:self.out_seq_length]
    

class RandomSequenceSortDatasetWithHoles(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=2, out_seq_length=1, ascending=True, range_left=-1, range_right=11, holes=[]):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.ascending = ascending
        self.range_left = range_left
        self.range_right = range_right
        self.holes = holes  # contains tuples of the form (left, right) which means no points should be sampled in the interval [left, right]
        self.valid_ranges = self._get_valid_ranges()

    def __len__(self):
        return self.num_samples  # Number of samples

    def _get_valid_ranges(self):
        # Start with a single range from range_left to range_right
        valid_ranges = [(self.range_left, self.range_right)]
        
        # Subtract the holes from this range
        for (left, right) in self.holes:
            new_ranges = []
            for (valid_left, valid_right) in valid_ranges:
                # If the hole is inside the valid range, split the range
                if left > valid_left and right < valid_right:
                    new_ranges.append((valid_left, left))
                    new_ranges.append((right, valid_right))
                elif left > valid_left:
                    new_ranges.append((valid_left, min(left, valid_right)))
                elif right < valid_right:
                    new_ranges.append((max(valid_left, right), valid_right))
                # If the hole is outside the valid range, keep the range unchanged
                else:
                    new_ranges.append((valid_left, valid_right))
            valid_ranges = new_ranges
            
        return valid_ranges

    def _sample_from_valid_ranges(self):
        # Choose a random range based on its length
        lengths = np.array([right - left for left, right in self.valid_ranges])
        probabilities = lengths / lengths.sum()
        selected_range = np.random.choice(len(self.valid_ranges), p=probabilities)
        
        # Sample from the selected range
        left, right = self.valid_ranges[selected_range]
        return torch.rand(1).item() * (right - left) + left

    def __getitem__(self, idx):
        # Sample points avoiding the holes
        x = torch.tensor([self._sample_from_valid_ranges() for _ in range(self.in_seq_length)])
        
        # Sort the sequence
        y, _ = torch.sort(x, descending=not self.ascending)

        # Select the first 'out_seq_length' elements
        y = y[:self.out_seq_length]

        return x.unsqueeze(1), y.unsqueeze(1)

class RandomSequenceMedianRankDataset(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=2, range_left=-1, range_right=11):
        assert in_seq_length % 2 == 1, "Input sequence length must be odd for a unique median."
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.range_left = range_left
        self.range_right = range_right
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left 
        x = x.unsqueeze(1)
        # find the position of the median in the sequence
        median = x.median()
        # if there are multiple medians, take the first one
        median_rank = torch.where(x == median)[0]
        # check if no median was found
        if median_rank.dim() == 0:
            print(f"No median found: {x}")
            print("Trying again")
            return self.__getitem__(idx)
        if len(median_rank) > 1:
            print(f"Multiple medians found: {median_rank}, {median_rank.shape}, {x}")
            median_rank = median_rank[0].unsqueeze(0)
        return x, median_rank.unsqueeze(1).float()
    
class RandomSequenceRankClosest(Dataset):
        # rank of closest to the value in position 0 (except 0)
    def __init__(self, num_samples=100_000, in_seq_length=2, range_left=-1, range_right=11):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.range_left = range_left
        self.range_right = range_right
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        # Generate a random sequence
        x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left
        x = x.unsqueeze(1)  # Reshape for any further operations needing 2D tensors
        
        # Calculate distances from the first element
        distances = torch.abs(x - x[0])
        
        # Find the index of the element closest to the first element, ignoring the first element itself
        distances[0] = float('inf')  # Set the first element's distance to infinity so it's not chosen
        closest_rank = torch.argmin(distances)  # Get the index of the smallest distance
        
        return x, closest_rank.unsqueeze(0).float()


class RandomSequenceSumDataset(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=2, range_left=-1, range_right=11):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.range_left = range_left
        self.range_right = range_right
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        # Generate a random sequence x
        x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left 
        x = x.unsqueeze(1)
        # Calculate the sum of the sequence
        y = torch.sum(x, dim=0)
        return x, y.unsqueeze(-1)
    

# class RandomSequenceKernelDataset(Dataset):
#     def __init__(self, num_samples=100_000, in_seq_length=2, range_left=-1, range_right=11, kernel="gaussian", length_scale=1):
#         self.num_samples = num_samples
#         self.in_seq_length = in_seq_length
#         self.range_left = range_left
#         self.range_right = range_right
#         self.kernel = kernel
#         self.length_scale = length_scale
        
#     def __len__(self):
#         return self.num_samples  # Number of samples

#     def __getitem__(self, idx):
#         # Generate a random sequence x
#         x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left 
#         x = x.unsqueeze(1)
#         # Calculate the kernel
#         if self.kernel == "gaussian":
#             y = torch.exp(-0.5 * (x - x.T) ** 2 / self.length_scale ** 2)[0, 1]
#         elif self.kernel == "laplacian":
#             y = torch.exp(-torch.abs(x - x.T) / self.length_scale)[0, 1]
#         elif self.kernel == "cauchy":
#             y = 1 / (1 + ((x - x.T) / self.length_scale) ** 2)[0, 1]
#         else:
#             raise ValueError("Unknown kernel")

#         return x, y

class RandomSequenceKernelDataset(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=10, range_left=-1, range_right=11, kernel="gaussian", length_scale=1):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.range_left = range_left
        self.range_right = range_right
        self.kernel = kernel
        self.length_scale = length_scale
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        # Generate a random sequence x
        x = torch.rand(self.in_seq_length) * (self.range_right - self.range_left) + self.range_left 
        x = x.unsqueeze(1)
        # compute the mean of the sequence, weighted by the kernel similarity to the first element
        if self.kernel == "gaussian":
            weights = torch.exp(-0.5 * (x - x[0]) ** 2 / self.length_scale ** 2)
            y = torch.sum(weights * x) / torch.sum(weights)
        elif self.kernel == "laplacian":
            weights = torch.exp(-torch.abs(x - x[0]) / self.length_scale)
            y = torch.sum(weights * x) / torch.sum(weights)
        elif self.kernel == "cauchy":
            weights = 1 / (1 + ((x - x[0]) / self.length_scale) ** 2)
            y = torch.sum(weights * x) / torch.sum(weights)
        else:
            raise ValueError("Unknown kernel")

        return x, y.unsqueeze(-1).unsqueeze(-1)
        

def generate_polynomial_data(batch_size, max_degree, coeff_range):
    """
    Generate a batch of polynomial coefficients and their roots for training.

    Parameters:
    - batch_size (int): Number of samples in the batch.
    - max_degree (int): Maximum degree of the polynomial.
    - coeff_range (tuple): Min and max range of coefficients (inclusive).
    
    Returns:
    - torch.Tensor: Batch of polynomial coefficients, padded with zeros.
    - torch.Tensor: Batch of polynomial roots, also padded with zeros.
    """
    coeffs_list = []
    roots_list = []

    for _ in range(batch_size):
        # Randomly choose the degree of the polynomial (at least 1)
        degree = np.random.randint(1, max_degree + 1)

        # Generate random coefficients for a polynomial of the chosen degree
        coeffs = np.random.uniform(coeff_range[0], coeff_range[1], degree + 1)

        # Ensure the leading coefficient is not zero
        coeffs[0] = np.random.uniform(max(coeff_range[0], 1e-6), coeff_range[1])

        # Pad the coefficients with zeros if there are less than max_degree + 1 coefficients
        padded_coeffs = np.pad(coeffs, (0, max_degree + 1 - len(coeffs)), 'constant', constant_values=(0,))

        # Find the roots of the polynomial
        roots = np.roots(coeffs)

        # Filter out complex roots, if necessary
        real_roots = roots[np.isreal(roots)].real

        # Pad the real roots with zeros if there are less than max_degree roots
        padded_roots = np.pad(real_roots, (0, max_degree - len(real_roots)), 'constant', constant_values=(0,))

        coeffs_list.append(torch.tensor(padded_coeffs, dtype=torch.float32))
        roots_list.append(torch.tensor(padded_roots, dtype=torch.float32))

    # Stack the lists to create batch tensors
    coeffs_tensor = torch.stack(coeffs_list)
    roots_tensor = torch.stack(roots_list)

    return coeffs_tensor, roots_tensor

class RandomSequencePolynomialDataset(Dataset):
    def __init__(self, num_samples=100_000, in_seq_length=2, range_left=-1, range_right=11):
        self.num_samples = num_samples
        self.in_seq_length = in_seq_length
        self.out_seq_length = in_seq_length - 1
        self.range_left = range_left
        self.range_right = range_right
        
    def __len__(self):
        return self.num_samples  # Number of samples

    def __getitem__(self, idx):
        x, y = generate_polynomial_data(1, self.in_seq_length - 1, (self.range_left, self.range_right))
        x = x.squeeze(0).unsqueeze(-1)
        y = y.squeeze(0).unsqueeze(-1)
        return x, y

        
    
# class RandomSequenceRankDatasetTrain(Dataset):
#     def __init__(self, num_samples=100_000):
#         self.num_samples = num_samples
        
#     def __len__(self):
#         return self.num_samples  # Number of samples

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 12 - 1 # Random numbers between -1 and 11
#         x = x.unsqueeze(1)
#         y = torch.argsort(x, dim=0).float()[0, :] #TODO: make classif
#         return x, y

# class RandomSequenceRankDatasetTest(Dataset):
#     def __len__(self):
#         return 10_000

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
#         x = x.unsqueeze(1)
#         y = torch.argsort(x, dim=0).float()[0, :]
#         return x, y

# # same for addition
# class RandomSequenceAdditionDatasetTrain(Dataset):
#     def __init__(self, num_samples=100_000):
#         self.num_samples = num_samples
        
#     def __len__(self):
#         return self.num_samples  # Number of samples

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 12 - 1 # Random numbers between -1 and 11
#         x = x.unsqueeze(1)
#         # sum
#         y = torch.sum(x).unsqueeze(0)
#         return x, y

# class RandomSequenceAdditionDatasetTest(Dataset):
#     def __len__(self):
#         return 10_000

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
#         x = x.unsqueeze(1)
#         y = torch.sum(x).unsqueeze(0)
#         return x, y

# # same for multiplication
# class RandomSequenceMultiplicationDatasetTrain(Dataset):
#     def __init__(self, num_samples=100_000):
#         self.num_samples = num_samples

#     def __len__(self):
#         return self.num_samples  # Number of samples

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 12 - 1 # Random numbers between -1 and 11
#         x = x.unsqueeze(1)
#         # sum
#         y = torch.prod(x).unsqueeze(0)
#         return x, y

# class RandomSequenceMultiplicationDatasetTest(Dataset):
#     def __len__(self):
#         return 10_000

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
#         x = x.unsqueeze(1)
#         y = torch.prod(x).unsqueeze(0)
#         return x, y

# # same for a weird function
# class RandomSequenceWeirdDatasetTrain(Dataset):
#     def __init__(self, num_samples=100_000):
#         self.num_samples = num_samples
        
#     def __len__(self):
#         return self.num_samples  # Number of samples

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 12 - 1 # Random numbers between -1 and 11
#         x = x.unsqueeze(1)
#         y = torch.log(torch.abs(x[0])) - torch.sin(x[1])
#         y = y.unsqueeze(0)
#         return x, y

# class RandomSequenceWeirdDatasetTest(Dataset):
#     def __len__(self):
#         return 10_000

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
#         x = x.unsqueeze(1)
#         y = torch.log(torch.abs(x[0]) + 1) - torch.sin(x[1])
#         y = y.unsqueeze(0)
#         return x, y
    
# class RandomSequenceSinDatasetTrain(Dataset):
#     def __init__(self, num_samples=100_000):
#         self.num_samples = num_samples
        
#     def __len__(self):
#         return self.num_samples  # Number of samples

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 12 - 1 # Random numbers between -1 and 11
#         x = x.unsqueeze(1)
#         y = torch.sin(torch.sum(x)).unsqueeze(0)
#         return x, y

# class RandomSequenceSinDatasetTest(Dataset):
#     def __len__(self):
#         return 10_000

#     def __getitem__(self, idx):
#         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
#         x = x.unsqueeze(1)
#         y = torch.sin(torch.sum(x)).unsqueeze(0)
#         return x, y
    

# # # same for subtraction
# # class RandomSequenceSubtractionDatasetTrain(Dataset):
# #     def __len__(self):
# #         return 100_000  # Number of samples

# #     def __getitem__(self, idx):
# #         x = torch.rand(seq_length) * 12 - 1 # Random numbers between -1 and 11
# #         x = x.unsqueeze(1)
# #         y = torch.zeros_like(x)
# #         # sum
# #         y = torch.zeros_like
# #         return x, y

# # class RandomSequenceSubtractionDatasetTest(Dataset):
# #     def __len__(self):
# #         return 10_000

# #     def __getitem__(self, idx):
# #         x = torch.rand(seq_length) * 10 # Random numbers between 0 and 10
# #         x = x.unsqueeze(1)
# #         y = torch.zeros_like(x)
# #         # sum
# #         y[0, :] = x[0, :] - x[1, :]
# #         return x, y
