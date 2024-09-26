import torch
def deactivate_layer_norm(model):
    #Specific to the current model
    for name, param in model.named_parameters():
        if "LayerNorm" in name:
            param.requires_grad = False
            if "bias" in name:
                param.data = torch.zeros_like(param.data)
            if "weight" in name:
                param.data = torch.ones_like(param.data)

def deactivate_positional_encoding(model):
    #Specific to the current model
    model._backbone.embeddings.position_embeddings.weight.data = torch.zeros_like(model._backbone.embeddings.position_embeddings.weight.data)
    model._backbone.embeddings.position_embeddings.weight.requires_grad = False