import copy

from prettytable import PrettyTable
from torch import Tensor, nn
import torch


def concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def stack_layers(module: nn.Module, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def count_parameters(model: nn.Module) -> tuple[PrettyTable, int]:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return table, total_params


def create_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """
    Create a causal (lower triangular) mask for self-attention.
    Returns True for positions that should be masked (ignored).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_padding_mask(seq: Tensor, pad_token_id: int = 0) -> Tensor:
    """Create padding mask where True indicates padding tokens."""
    return seq == pad_token_id


def make_tgt_mask(tgt: Tensor, pad_id: int) -> Tensor:
    "Create a mask to hide padding and future words."
    # unsqueeze to (16,1,128)
    tgt_mask = (tgt == pad_id).unsqueeze(-2)
    # print(f"padding mask dims {tgt_mask.shape}")
    # plt.imsave("padding_mask.png", tgt_mask.squeeze(1))
    # unsqueeze to (1,128,128)
    m = torch.full((tgt.shape[-1], tgt.shape[-1]), True, device=tgt.device, dtype=torch.bool)
    triangle_mask = torch.triu(m, diagonal=1).unsqueeze(0)
    # print(f"triangle mask dimes {triangle_mask.shape}")
    # plt.imsave("triangle_mask.png", triangle_mask[0])
    # unsqueeze to (16,1,128,128)
    tgt_mask = (tgt_mask | triangle_mask).unsqueeze(1)
    # plt.imsave("combined_mask.png", tgt_mask[0][0])
    # print(f"tgt mask dims {tgt_mask.shape}")
    return tgt_mask
