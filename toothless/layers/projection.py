import torch
from torch import nn, Tensor


class InputProjection(nn.Module):
    def __init__(
        self,
        E_q: int,
        E_total: int,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


class PackedInputProjection(nn.Module):
    def __init__(
        self,
        E_q: int,
        E_total: int,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q_proj, k_proj, v_proj = torch.chunk(self.packed_proj(query), 3, dim=-1)
        return q_proj, k_proj, v_proj
