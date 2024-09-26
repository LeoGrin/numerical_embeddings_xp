import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncodingTabular(nn.Module):
    def __init__(self, d_model, sigma=1.0):
        super(PositionalEncodingTabular, self).__init__()
        self.d_model = d_model
        # sample frequencies from a normal distribution
        self.freqs = nn.Parameter(torch.randn(d_model) * sigma)

    def forward(self, x):
        # Get batch size, sequence length, and device
        batch_size, seq_length, _ = x.shape
        try:
            device = x.device
        except:
            device = torch.device('cpu')
        
        x = x.squeeze(-1)

        # Calculate the positional encoding
        pos_enc = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        for i in range(0, self.d_model, 2):
            div_term = 2 * np.pi * self.freqs[i]
            pos_enc[:, :, i] = torch.sin(x * div_term)
            pos_enc[:, :, i + 1] = torch.cos(x * div_term)

        return pos_enc

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_inv_freq=10000.0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_inv_freq = max_inv_freq

    def forward(self, x):
        # Get batch size, sequence length, and device
        batch_size, seq_length, _ = x.shape
        try:
            device = x.device
        except:
            device = torch.device('cpu')
        
        x = x.squeeze(-1)

        # Calculate the positional encoding
        pos_enc = torch.zeros(batch_size, seq_length, self.d_model, device=device)
        for i in range(0, self.d_model, 2):
            div_term = torch.exp(i * -(torch.log(torch.tensor(self.max_inv_freq, device=device)) / self.d_model))
            pos_enc[:, :, i] = torch.sin(x * div_term)
            pos_enc[:, :, i + 1] = torch.cos(x * div_term)

        return pos_enc
    
    def revert(self, encoding, verbose=0):
        d_encoding = encoding.shape[2]
        mask = (encoding <= 1.0) & (encoding >= -1.0)
        print("mask", mask.sum(), mask.numel())
        if torch.rand(1) < 0.01:
            print("masked proportion", torch.sum(mask) / (encoding.shape[0] * encoding.shape[1] * encoding.shape[2]))
        encoding_masked = encoding * mask
        estimation = [torch.arcsin(encoding_masked[:, :, i]) / self.max_inv_freq**(-i / d_encoding) for i in range(0, d_encoding, 2)] + \
            [torch.arccos(encoding_masked[:, :, i]) / self.max_inv_freq**(-i / d_encoding) for i in range(1, d_encoding, 2)]
        print(estimation)
        # compute nan mean
        estimation = torch.stack(estimation, dim=2)
        # use cat
        #estimation = torch.cat(estimation, dim=1)
        if verbose:
            print(estimation)
        #estimation_mean = torch.nanmean(estimation, dim=2)
        #estimation_mean = torch.mean(masked_tensor(estimation, ~torch.isnan(estimation)), dim=2)
        #estimation_mean = torch.mean(estimation, dim=1).unsqueeze(1)
        #estimation_mean = torch.zeros(estimation.shape[0], estimation.shape[1], 1, device=encoding.device)
        #for i in range(estimation.shape[0]):
        #    for j in range(estimation.shape[1]):
        #        estimation_mean[i, j, 0] = torch.mean(torch.masked_select(estimation[i, j, :], ~torch.isnan(estimation[i, j, :])))
        return torch.sum(estimation * mask, dim=2) / torch.sum(mask, dim=2)
    
    def revert_exp(self, encoding, verbose=0):
        d_encoding = encoding.shape[2]
        mask = (encoding <= 1.0) & (encoding >= -1.0)
        print("mask", mask.sum(), mask.numel())
        if torch.rand(1) < 0.01:
            print("masked proportion", torch.sum(mask) / (encoding.shape[0] * encoding.shape[1] * encoding.shape[2]))
        encoding_masked = encoding * mask
        estimation = [torch.arcsin(encoding_masked[:, :, i]) / self.max_inv_freq**(-i / d_encoding) for i in range(0, d_encoding, 2)] + \
            [torch.arccos(encoding_masked[:, :, i]) / self.max_inv_freq**(-i / d_encoding) for i in range(1, d_encoding, 2)]
        # bin by tol and take the majority
        with torch.no_grad():
            tol = 1e-1
            estimation = torch.stack(estimation, dim=2)
            print("estimation", estimation.shape)
            estimation_binned = torch.round(estimation / tol) * tol
            estimation_mode = torch.mode(estimation_binned, dim=2).values.unsqueeze(2)
            print("estimation_mode", estimation_mode.shape)
            print("estimation_binned", estimation_binned.shape)
            # only keep the values that are close to the mode
            mask_mode = (estimation_binned - estimation_mode).abs() < tol
            print("mask_mode", mask_mode.shape)
            print("mask_mode", mask_mode.sum(), mask_mode.numel())
        #print(estimation)
        # compute nan mean
        # use cat
        #estimation = torch.cat(estimation, dim=1)
        if verbose:
            print(estimation)
        #estimation_mean = torch.nanmean(estimation, dim=2)
        #estimation_mean = torch.mean(masked_tensor(estimation, ~torch.isnan(estimation)), dim=2)
        #estimation_mean = torch.mean(estimation, dim=1).unsqueeze(1)
        #estimation_mean = torch.zeros(estimation.shape[0], estimation.shape[1], 1, device=encoding.device)
        #for i in range(estimation.shape[0]):
        #    for j in range(estimation.shape[1]):
        #        estimation_mean[i, j, 0] = torch.mean(torch.masked_select(estimation[i, j, :], ~torch.isnan(estimation[i, j, :])))
        #return estimation
        return torch.sum(estimation * mask_mode, dim=2) / torch.sum(mask_mode, dim=2)


# class GaussianEmbedder(nn.Module):
#     def __init__(self, embed_dim, std_dev, left_bound=-2, right_bound=13):
#         super(GaussianEmbedder, self).__init__()
#         self.embed_dim = embed_dim
#         self.std_dev = std_dev
#         self.left_bound = left_bound
#         self.right_bound = right_bound

#         # Generate the grid for embedding dimensionsl
#         self.register_buffer('embed_grid', torch.linspace(left_bound, right_bound, embed_dim).view(1, 1, embed_dim))
#     def forward(self, x):
#         batch_size, seq_length, _ = x.shape
#         # Expand x to (batch_size, seq_length, embed_dim) by repeating along the new dimension
#         x_expanded = x.repeat(1, 1, self.embed_dim)
#         embed_grid = self.embed_grid.repeat(batch_size, seq_length, 1)
#         # Compute the Gaussian embedding
#         gaussian_embed = torch.exp(-torch.pow(x_expanded - embed_grid, 2) / (2 * self.std_dev ** 2))
#         return gaussian_embed

#     def revert(self, gaussian_embed):
#         batch_size, seq_length, embed_dim = gaussian_embed.shape
#         assert embed_dim == self.embed_dim, "Embedding dimension mismatch."

#         gaussian_embed = torch.abs(gaussian_embed)

#         # Create the grid for embedding dimensions, similar to the forward method
#         embed_grid = self.embed_grid.repeat(batch_size, seq_length, 1).to(gaussian_embed.device)

#         # Calculate the weighted sum of the grid positions based on the embedding weights
#         weighted_positions = gaussian_embed * embed_grid
        
#         # Sum over the embedding dimension and normalize by the sum of the weights to get the mean
#         position_sums = weighted_positions.sum(dim=2)
#         weight_sums = gaussian_embed.sum(dim=2)
#         #print("weighted_positions", weighted_positions)
        
#         # Avoid division by zero in case of very small weight_sums
#         weight_sums = weight_sums.clamp(min=1e-10)
        
#         # Calculate the mean position (this is the approximate inversion of the embedding)
#         mean_positions = position_sums / weight_sums
        
#         return mean_positions.unsqueeze(-1)
    
class GaussianEmbedder(nn.Module):
    def __init__(self, embed_dim, std_dev_list, left_bound=-2, right_bound=13, parametrize_std=False,
                 embed_grid = "equally_spaced", normalization="step", scale=1.0, parametrize_scale=False):
        super(GaussianEmbedder, self).__init__()
        self.embed_dim = embed_dim
        self.std_dev_list = std_dev_list
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.step_size = torch.tensor((right_bound - left_bound) / embed_dim) #TODO: check if this is correct when I have multiple std devs
        self.normalization = normalization

        if parametrize_std:
            #self.std_dev = nn.Parameter(torch.tensor(std_dev).float())
            self.std_dev_list = nn.ParameterList([nn.Parameter(torch.tensor(std_dev).float()) for std_dev in std_dev_list])

        # check that the embedding dim is divisible by the number of std devs
        assert embed_dim % len(std_dev_list) == 0, "Embedding dimension must be divisible by the number of standard deviations."

        # Generate the grid for embedding dimensions
        if embed_grid == "equally_spaced":
            #TODO: we don't need to make a grid for each standard deviation
            self.register_buffer('embed_grid', torch.stack([torch.linspace(left_bound, right_bound, embed_dim // len(std_dev_list)).view(1, 1, embed_dim // len(std_dev_list)) for _ in std_dev_list], dim=0))
        elif embed_grid == "uniform":
            # make a different grid for each standard deviation
            self.register_buffer('embed_grid', torch.stack([torch.linspace(left_bound, right_bound, embed_dim // len(std_dev_list)).view(1, 1, embed_dim // len(std_dev_list)) for _ in std_dev_list], dim=0))
        elif embed_grid == "normal":
            # Generate a grid using normal distribution for each standard deviation
            mean = 0#(left_bound + right_bound) / 2.0
            self.register_buffer('embed_grid', torch.stack([
                torch.normal(mean, 1., (1, 1, embed_dim // len(std_dev_list)))
                for _ in std_dev_list], dim=0))
        else:
            raise ValueError("Invalid embed_grid argument. Must be one of 'equally_spaced', 'uniform', or 'normal'.")

        if parametrize_scale:
            self.scale = nn.Parameter(torch.tensor(scale).float())
        else:
            self.scale = scale

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        # Expand x to (batch_size, seq_length, embed_dim) by repeating along the new dimension
        x_expanded = x.repeat(1, 1, self.embed_dim // len(self.std_dev_list))
        embed_grid = self.embed_grid.repeat(1, batch_size, seq_length, 1)
        # Compute the Gaussian embedding
        #gaussian_embed = torch.exp(-torch.pow(x_expanded - embed_grid, 2) / (2 * self.std_dev ** 2))
        gaussian_embed = torch.cat([torch.exp(-torch.pow(x_expanded - embed_grid[i], 2) / (2 * self.std_dev_list[i] ** 2)) for i in range(len(self.std_dev_list))], dim=2)
        if self.normalization == "step":
            output = gaussian_embed * torch.sqrt(self.step_size) #TODO: think whether we should just normalize to norm 1
        elif self.normalization == "norm":
            output =  gaussian_embed / torch.norm(gaussian_embed, dim=2, keepdim=True)
        else:
            output =  gaussian_embed
        
        return output * self.scale

    def revert(self, gaussian_embed):
        assert self.scale == 1.0, "Reversion is only implemented for scale=1.0."
        batch_size, seq_length, embed_dim = gaussian_embed.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch."

        gaussian_embed = torch.abs(gaussian_embed)

        # Create the grid for embedding dimensions, similar to the forward method
        #embed_grid = self.embed_grid.repeat(batch_size, seq_length, 1).to(gaussian_embed.device)
        embed_grid = self.embed_grid.reshape(1, 1, self.embed_dim).repeat(batch_size, seq_length, 1).to(gaussian_embed.device)

        # Calculate the weighted sum of the grid positions based on the embedding weights
        weighted_positions = gaussian_embed * embed_grid
        
        # Sum over the embedding dimension and normalize by the sum of the weights to get the mean
        position_sums = weighted_positions.sum(dim=2)
        weight_sums = gaussian_embed.sum(dim=2)
        #print("weighted_positions", weighted_positions)
        
        # Avoid division by zero in case of very small weight_sums
        weight_sums = weight_sums.clamp(min=1e-10)
        
        # Calculate the mean position (this is the approximate inversion of the embedding)
        mean_positions = position_sums / weight_sums
        
        return mean_positions.unsqueeze(-1)


# class RationalQuadraticKernelDensity(dist.TorchDistribution):
#     """
#     A custom Pyro distribution for the spectral density derived from the rational quadratic kernel's Fourier transform.
#     """
#     def __init__(self, alpha, length_scale, validate_args=None):
#         self.alpha = alpha
#         self.length_scale = length_scale
#         super().__init__(validate_args=validate_args)
    
#     def log_prob(self, k):
#         """
#         Compute log probability density function based on derived Fourier transform.
#         """
#         coefficient = 2**(1.5 - self.alpha) * (1 / (self.alpha * self.length_scale**2))**(-0.25 - self.alpha / 2) * np.sqrt(np.pi)
#         gamma_term = gamma(self.alpha)
#         bessel_term = kv(0.5 - self.alpha, np.abs(k) / np.sqrt(1 / (self.alpha * self.length_scale**2)))
#         density = coefficient * np.abs(k)**(-0.5 + self.alpha) * bessel_term / gamma_term
#         return torch.log(density)

class RandomFeatureEmbedder(nn.Module):
    def __init__(self, num_random_features, kernel='gaussian', alpha=1.0, length_scale=1.0, scale=1.0):
        super(RandomFeatureEmbedder, self).__init__()
        input_dim = 1  # Assuming input features are 1-dimensional
        self.num_random_features = num_random_features
        self.kernel = kernel
        self.length_scale = length_scale
        self.alpha = alpha

        if kernel == 'gaussian':
            # Fourier transform of the Gaussian kernel: exp(-0.5 * (length_scale * omega)^2)
            self.weights = nn.Parameter(torch.randn(num_random_features, input_dim) / length_scale)
        elif kernel == 'laplacian':
            # Fourier transform of the Laplacian kernel: 1 / (1 + (length_scale * omega)^2)
            self.weights = nn.Parameter(torch.from_numpy(np.random.standard_cauchy(size=(num_random_features, input_dim)).astype(np.float32)) / length_scale)
        elif kernel == 'cauchy':
            # Fourier transform of the Cauchy kernel: exp(-length_scale * |omega|)
            self.weights = nn.Parameter(torch.from_numpy(np.random.laplace(loc=0, scale=1.0/length_scale, size=(num_random_features, input_dim)).astype(np.float32)))
        elif kernel == 'rational_quadratic':
            # # the fourier transform is ...
            # # we use pyro to sample from the distribution
            # spectral_density = RationalQuadraticKernelDensity(alpha, length_scale)
            # # Set up the MCMC runner with a basic kernel
            # #nuts_kernel = NUTS(lambda: pyro.sample("omega", spectral_density))
            # #hmc_kernel = HMC(lambda: pyro.sample("omega", spectral_density), step_size=0.0855, num_steps=4, full_mass=False)
            # mcmc_kernel = MCMCKernel(spectral_density)

            # mcmc = MCMC(mcmc_kernel, num_samples=num_random_features, warmup_steps=200)

            # # Perform MCMC sampling
            # mcmc.run()
            
            # samples = mcmc.get_samples()["omega"]
            # self.weights = nn.Parameter(samples)
            raise NotImplementedError

        else:
            raise ValueError("Unsupported kernel type. Supported types: 'gaussian', 'laplacian', 'cauchy', 'rational_quadratic'")

        self.biases = nn.Parameter(torch.rand(num_random_features) * 2 * np.pi)
        self.scale = scale

    def forward(self, x):
        projected = F.linear(x, self.weights) + self.biases
        features = torch.cos(projected) * np.sqrt(2.0 / self.num_random_features)
        return features * self.scale
    
#TODO
class RandomBinningFeatures(nn.Module):
    def __init__(self, P, distribution_sampler, input_range=(0, 1)):
        """
        Args:
            P (int): The number of random grids (features).
            distribution_sampler (callable): A function that samples δ from the desired distribution p(∆).
            input_range (tuple): The range (min, max) of the input feature values.
        """
        super(RandomBinningFeatures, self).__init__()
        self.P = P
        self.distribution_sampler = distribution_sampler
        self.input_range = input_range
        self.register_buffer('grids', self._create_random_grids())

    def _create_random_grids(self):
        """
        Create P random grids with random pitch and shifts drawn from the specified distributions.
        """
        delta = self.distribution_sampler(self.P)
        u = torch.rand(self.P) * delta
        grids = torch.stack([delta, u], dim=1)
        return grids

    def _hash_function(self, x, grid):
        """
        Hash function that maps the input x to the index of the bin into which it falls.
        """
        delta, u = grid.unbind(-1)
        return torch.floor((x - u) / delta).long()

    def forward(self, x):
        """
        Transform the input x into the P-dimensional random binning feature space.
        """
        batch_size, seq_length = x.shape
        Z = torch.zeros(batch_size, seq_length, self.P, device=x.device)
        for i in range(self.P):
            Z[:, :, i] = self._hash_function(x, self.grids[i])
        return Z / torch.sqrt(torch.tensor(self.P, dtype=torch.float32, device=x.device))


class LinearEmbedder(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(LinearEmbedder, self).__init__()
        self.embed_dim = embed_dim
        self._read_in = nn.Linear(1, embed_dim, bias=bias)

    def forward(self, x):
        return self._read_in(x)

    def revert(self, linear_embed):
        raise NotImplementedError
    
import torch
import torch.nn as nn

class IndependentLinearEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim, bias=False):
        """
        Initialize the IndependentLinearEmbedder module.

        Args:
        input_dim (int): Number of independent input dimensions.
        embed_dim (int): The size of each embedding vector for each input dimension.
        bias (bool): Whether to add a bias term in each linear transformation.
        """
        super(IndependentLinearEmbedder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Creating a ModuleList of Linear layers, one for each input dimension
        self.linears = nn.ModuleList([
            nn.Linear(1, embed_dim, bias=bias) for _ in range(input_dim)
        ])

    def forward(self, x):
        """
        Forward pass through multiple linear transformations.

        Args:
        x (torch.Tensor): The input tensor of shape (batch_size, input_dim, 1)

        Returns:
        torch.Tensor: The output tensor of shape (batch_size, input_dim, embed_dim)
        """
        # Apply each linear layer to each corresponding dimension of the input.
        # x should be of shape (batch_size, input_dim), and we process each 'slice' of x independently.
        outputs = [linear(x[:, i, :]) for i, linear in enumerate(self.linears)]
        # Concatenate the outputs along the last dimension to form a single tensor
        return torch.cat(outputs, dim=-1)