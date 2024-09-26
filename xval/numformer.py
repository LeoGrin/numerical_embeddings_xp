import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
from embeddings import RandomFeatureEmbedder, GaussianEmbedder

class GaussianEmbedderOld(nn.Module):
    def __init__(self, embed_dim, std_dev, left_bound=-2, right_bound=13,
                 parametrize_std=False):
        super(GaussianEmbedderOld, self).__init__()
        self.embed_dim = embed_dim
        self.std_dev = std_dev
        self.left_bound = left_bound
        self.right_bound = right_bound

        if parametrize_std:
            self.std_dev = nn.Parameter(torch.tensor(std_dev).float())

        # Generate the grid for embedding dimensions
        self.register_buffer('embed_grid', torch.linspace(left_bound, right_bound, embed_dim).view(1, 1, embed_dim))
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        # Expand x to (batch_size, seq_length, embed_dim) by repeating along the new dimension
        x_expanded = x.repeat(1, 1, self.embed_dim)
        embed_grid = self.embed_grid.repeat(batch_size, seq_length, 1)
        # Compute the Gaussian embedding
        gaussian_embed = torch.exp(-torch.pow(x_expanded - embed_grid, 2) / (2 * self.std_dev ** 2))
        return gaussian_embed

    def revert(self, gaussian_embed):
        batch_size, seq_length, embed_dim = gaussian_embed.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch."

        gaussian_embed = torch.abs(gaussian_embed)

        # Create the grid for embedding dimensions, similar to the forward method
        embed_grid = self.embed_grid.repeat(batch_size, seq_length, 1).to(gaussian_embed.device)

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
    

        
class Numformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=6,
        num_layers=6,
        dim_feedforward=3072,
        dropout=0.1,
        activation=nn.GELU(),
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        transformer_bias=False,
        numhead_bias=True,
        context_length=1024,
        is_causal=False,
        numerical_embedding="fourier",
        gaussian_embedding_sd=[0.06],
        reverse_numerical_embedding=False,
        parametrize_std=False,
        gaussian_embedding_left_bound=-0.5,
        gaussian_embedding_right_bound=1.5,
        gaussian_embedding_normalization="step",
        gaussian_embedding_embed_grid="equally_spaced",
        fourier_length_scale=1.0,
        fourier_kernel="gaussian",
        normalize_pos_enc=True,
        normalize_token_emb=True,
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            # bias=transformer_bias,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder, num_layers=num_layers, enable_nested_tensor=False
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(context_length, d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )
        self.num_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )
        self.is_causal = is_causal
        self.numerical_embedding = numerical_embedding
        if self.numerical_embedding == "gaussian":
            self.numerical_embedder = GaussianEmbedder(
                embed_dim=d_model, std_dev_list=gaussian_embedding_sd, left_bound=gaussian_embedding_left_bound, right_bound=gaussian_embedding_right_bound, parametrize_std=parametrize_std,
                normalization=gaussian_embedding_normalization, embed_grid=gaussian_embedding_embed_grid
            ).float()
        elif self.numerical_embedding == "fourier":
            self.numerical_embedder = RandomFeatureEmbedder(
                num_random_features=d_model, kernel=fourier_kernel, length_scale=fourier_length_scale
            ).float()
        elif self.numerical_embedding == "gaussian_old":
            self.numerical_embedder = GaussianEmbedderOld(
                embed_dim=d_model, std_dev=gaussian_embedding_sd[0], left_bound=gaussian_embedding_left_bound, right_bound=gaussian_embedding_right_bound, parametrize_std=parametrize_std,
            ).float()
        self.reverse_numerical_embedding = reverse_numerical_embedding
        self.normalize_pos_enc = normalize_pos_enc
        self.normalize_token_emb = normalize_token_emb

    def forward(self, x, x_num):
        numerical_mask = x == 3
        positional_encoding = self.position_embed.weight[: x.shape[1]].unsqueeze(0)
        if self.normalize_pos_enc:
            positional_encoding = positional_encoding / torch.sqrt(
            torch.tensor(self.position_embed.embedding_dim).float()
        )
        x = self.token_embed(x)
        if self.normalize_token_emb:
            x = x / torch.sqrt(torch.tensor(self.token_embed.embedding_dim).float())
        if self.numerical_embedding == "linear":
            x = x * x_num.unsqueeze(-1)
            x = x + positional_encoding
        else:
            assert self.numerical_embedding in ["gaussian", "fourier", "gaussian_old"]
            x_num = self.numerical_embedder(x_num.unsqueeze(-1).float())
            # set the numerical embeddings to zero for the unmasked tokens
            #x_num[~numerical_mask] = 0
            zeros = torch.zeros_like(x_num)
            # Use where to perform the conditional assignment without in-place modification
            x_num = torch.where(numerical_mask.unsqueeze(-1), x_num, zeros)
            # print norm
            #print("x_num norm", x_num.norm(dim=-1))
            #print("x_num shape", x_num.shape)
            # remove the token embeddings for the masked tokens
            x[numerical_mask] = 0
            x = x + x_num
            # print positional_encoding norm
            #print("positional_encoding norm", positional_encoding.norm(dim=-1))
            #print("positional_encoding shape", positional_encoding.shape)
            x = x + positional_encoding

        x = self.encoder_stack(x, is_causal=self.is_causal)
        logit_preds = self.lm_head(x)
        if self.reverse_numerical_embedding:
            num_preds = self.numerical_embedder.revert(x)
        else:
            num_preds = self.num_head(x)
        return logit_preds, num_preds


### Define collator and data loaders
def define_masked_num_collator(pad_token_id, mask_token_id, mlm_probability):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        probability_matrix = torch.full(x.shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        y = x.clone()
        y_num = x_num.clone()
        y[~mask] = -100
        x[mask] = mask_token_id
        x_num[mask] = 1
        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}

    return masked_num_collator

def define_masked_num_collator_last(pad_token_id, mask_token_id):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        #probability_matrix = torch.full(x.shape, mlm_probability)
        #mask = torch.bernoulli(probability_matrix).bool()
        # mask last token
        mask = torch.zeros_like(x).bool()
        mask[torch.arange(x.shape[0]), torch.tensor([len(sample["input_ids"]) - 1 for sample in batch])] = True
        y = x.clone()
        y_num = x_num.clone()
        y[~mask] = -100
        x[mask] = mask_token_id
        x_num[mask] = 1
        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}

    return masked_num_collator


def define_masked_num_collator_one(pad_token_id, mask_token_id, mask_only_numbers=False):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        
        # Initialize the mask tensor with all False (no token is masked initially)
        mask = torch.zeros_like(x, dtype=torch.bool)
        
        # For each sample in the batch, select one random index to mask
        batch_size, seq_length = x.size()
        for i in range(batch_size):
            if mask_only_numbers:
                # Find indices where the token is a number (ID 3) and not a padding token
                valid_indices = ((x[i] == 3) & (x[i] != pad_token_id)).nonzero(as_tuple=True)[0]
            else:
                # Any token can be masked except the padding token
                valid_indices = (x[i] != pad_token_id).nonzero(as_tuple=True)[0]

            if len(valid_indices) > 0:
                # Randomly choose one index to mask
                masked_index = valid_indices[torch.randint(len(valid_indices), (1,))]
                mask[i, masked_index] = True
        
        y = x.clone()
        y_num = x_num.clone()
        
        # Apply the mask: Set masked token labels in `y` to -100 to ignore in loss computation
        y[~mask] = -100
        
        # Replace the input token with `mask_token_id` at the masked position
        x[mask] = mask_token_id
        
        # Similarly for numeric inputs, set the masked numeric input to a default value (1 here)
        x_num[mask] = 1

        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}

    return masked_num_collator

