from transformers import GPT2Model, GPT2Config, BertConfig, BertModel
import torch
from torch import nn
from typing import Optional, Tuple
import math

class BertSelfAttentionTemperature(nn.Module):
    def __init__(self, config, position_embedding_type=None, temperature=1.0):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.temperature = temperature

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        #attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # temperature
        attention_probs = nn.functional.softmax(attention_scores / self.temperature, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

# make a pytorch module which takes two inputs and sums them
class Adder(nn.Module):
    def __init__(self):
        super(Adder, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(1, 1))
    def forward(self, x, y):
        return self.W * x + y


class TransformerModel(nn.Module):
    def __init__(self, in_seq_length, n_embd=128, n_layer=12, n_head=4,
                 attention_only=False, softmax_temperature=1.0, d_out=1,
                out_seq_length=None, verbose=0, read_in_bias=False,
                only_one_emb=True):
        super(TransformerModel, self).__init__()
        # configuration = GPT2Config(
        #     n_positions=2 * n_positions,
        #     n_embd=n_embd,
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     resid_pdrop=0.0,
        #     embd_pdrop=0.0,
        #     attn_pdrop=0.0,
        #     use_cache=False,
        # )
        # self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        configuration = BertConfig(
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=4 * n_embd,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=2 * in_seq_length,
            type_vocab_size=1,
            initializer_range=0.02,
        )

        if out_seq_length is None:
            self.out_seq_length = in_seq_length
        else:
            self.out_seq_length = out_seq_length
        self.n_positions = in_seq_length
        self.n_dims = in_seq_length
        self.read_in_bias = read_in_bias
        self.only_one_emb = only_one_emb
        if only_one_emb:
            self._read_in = nn.Linear(1, n_embd, bias=read_in_bias)
        else:
            self._read_in = nn.Linear(in_seq_length, n_embd, bias=read_in_bias)
        #self._backbone = BertModel(configuration, add_pooling_layer=False)
        self._backbone = BertModel(configuration)
        if attention_only:
            self.attention_only = True
            # the model should be attention only
            # set all mlps to identity and require_grad=False
            for i in range(n_layer):
                self._backbone.encoder.layer[i].intermediate = nn.Identity()
                # self._backbone.encoder.layer[i].output should take two inputs and sum them
                self._backbone.encoder.layer[i].output = Adder()
                #self._backbone.encoder.layer[i].attention.output = Adder() #TODO
        if softmax_temperature != 1.0:
            # replace the attention module with a new one with a different softmax temperature
            for i in range(n_layer):
                self._backbone.encoder.layer[i].attention.self = BertSelfAttentionTemperature(
                    configuration,
                    temperature=softmax_temperature,
                )
        self._read_out = nn.Linear(n_embd, d_out)
        self.verbose = verbose
        

    def forward(self, xs):
        embeds = self._read_in(xs)
        if self.verbose:
            print(f"embeds.shape={embeds.shape}")
            print(f"embeds={embeds}")
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        if self.verbose:
            print(f"output.shape={output.shape}")
            print(f"output={output}")

        prediction = self._read_out(output)
        # only keep the first out_seq_length elements
        prediction = prediction[:, :self.out_seq_length]

        return prediction
    
    

from model import BertModel, BertConfig, Adder, BertSelfAttentionTemperature
class TransformerModelWithEmbeddings(TransformerModel):
    def __init__(self, embedding_function, in_seq_length, n_embd=128, n_layer=12, n_head=4,
                 attention_only=False, softmax_temperature=1.0, d_out=1,
                out_seq_length=None, verbose=0, read_in_bias=False, only_one_emb=True, revert_embedding=False,
                output_attentions=False):
        super(TransformerModelWithEmbeddings, self).__init__(
            in_seq_length=in_seq_length, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            attention_only=attention_only, softmax_temperature=softmax_temperature, d_out=d_out,
            out_seq_length=out_seq_length, verbose=verbose, read_in_bias=read_in_bias, only_one_emb=only_one_emb,
        )
        self.embedding_function = embedding_function
        self.revert_embedding = revert_embedding
        if not revert_embedding:
            self._read_out = nn.Linear(n_embd, d_out)
        self.output_attentions = output_attentions
    def forward(self, xs):
        # xs should be of shape (batch_size, n_positions, n_embd)
        #embeds = self._read_in(xs)
        if self.embedding_function is not None:
            embeds = self.embedding_function(xs)
        else:
            embeds = self._read_in(xs)
        backbone_output = self._backbone(inputs_embeds=embeds, output_attentions=self.output_attentions)
        output = backbone_output.last_hidden_state
        #prediction = self._read_out(output)
        if not self.revert_embedding or self.embedding_function is None:
            prediction = self._read_out(output)
        else:
            prediction = self.embedding_function.revert(output)
        # only keep the first out_seq_length elements
        prediction = prediction[:, :self.out_seq_length]

        if self.output_attentions:
            return prediction, backbone_output.attentions
        return prediction
    



# TAKEN FROM https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns