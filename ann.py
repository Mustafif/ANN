# from enum import Enum

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Function

# # ── Chaotic activation functions ─────────────────────────────────────────────


# class _HeavisideFn(Function):
#     @staticmethod
#     def forward(ctx, x):
#         return (x > 0).to(x.dtype)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return torch.zeros_like(grad_output)


# class _DReLUFn(Function):
#     @staticmethod
#     def forward(ctx, x, a):
#         ctx.save_for_backward(x)
#         ctx.a = a
#         return torch.where(x >= a, x, torch.zeros_like(x))

#     @staticmethod
#     def backward(ctx, grad_output):
#         return torch.zeros_like(grad_output), None


# class _DExpFn(Function):
#     @staticmethod
#     def forward(ctx, x):
#         out = torch.where(x >= 0, torch.exp(x), torch.zeros_like(x))
#         ctx.save_for_backward(x, out)
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         return torch.zeros_like(grad_output)


# class Heaviside(nn.Module):
#     def forward(self, x):
#         return _HeavisideFn.apply(x)


# class DReLU(nn.Module):
#     def __init__(self, a: float = 0.2):
#         super().__init__()
#         self.a = a

#     def forward(self, x):
#         return _DReLUFn.apply(x, self.a)


# class DExp(nn.Module):
#     def forward(self, x):
#         return _DExpFn.apply(x)


# # ── Chaotic activation selector ───────────────────────────────────────────────


# class ChaoticAct(Enum):
#     HEAVISIDE = "heaviside"  # D1
#     DRELU = "drelu"  # D2
#     DEXP = "dexp"  # D3


# def build_chaotic_act(kind: ChaoticAct, drelu_a: float = 0.2) -> nn.Module:
#     if kind == ChaoticAct.HEAVISIDE:
#         return Heaviside()
#     if kind == ChaoticAct.DRELU:
#         return DReLU(a=drelu_a)
#     if kind == ChaoticAct.DEXP:
#         return DExp()
#     raise ValueError(f"Unknown chaotic activation: {kind}")


# # ── D-Layer ───────────────────────────────────────────────────────────────────


# class DLayer(nn.Module):
#     """
#     Hybrid D-layer: exactly matching the previous architecture.
#     """

#     def __init__(
#         self,
#         features: int,
#         chaotic_act: ChaoticAct = ChaoticAct.DRELU,
#         stable_act: nn.Module | None = None,
#         drelu_a: float = 0.2,
#         chaotic_ratio: float = 0.2,
#     ):
#         super().__init__()
#         if not 0.0 <= chaotic_ratio <= 1.0:
#             raise ValueError("chaotic_ratio must be in [0, 1]")

#         self.features = features
#         self.chaotic_ratio = chaotic_ratio
#         self.n_chaotic = int(features * chaotic_ratio)
#         self.n_stable = features - self.n_chaotic

#         self.stable_act = stable_act if stable_act is not None else nn.ReLU()
#         self.chaotic_act = build_chaotic_act(chaotic_act, drelu_a)

#         self.stable_linear = nn.Linear(features, features)
#         self.chaotic_linear = nn.Linear(self.n_chaotic, self.n_chaotic)

#         self.mu = nn.Parameter(torch.zeros(self.n_chaotic))
#         self.alpha = nn.Parameter(torch.ones(self.n_chaotic))
#         self.pre_ln = nn.LayerNorm(features)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         base_out = self.stable_linear(x)
#         base_out = self.pre_ln(base_out)
#         base_out = self.stable_act(base_out)

#         if self.n_chaotic == 0:
#             return base_out

#         stable_part = base_out[:, : self.n_stable]
#         chaotic_target_part = base_out[:, self.n_stable :]

#         chaos_in = self.chaotic_linear(chaotic_target_part)
#         raw_chaos = self.chaotic_act(chaos_in)
#         weighted_chaos = self.alpha * raw_chaos

#         d_neurons = self.mu + chaos_in + weighted_chaos

#         return torch.cat([stable_part, d_neurons], dim=1)


# # ── ForwardModel ──────────────────────────────────────────────────────────────


# class ForwardModel(nn.Module):
#     def __init__(
#         self,
#         input_features: int = 15,
#         hidden_size: int = 200,
#         dropout_rate: float = 0.0,
#         num_layers: int = 5,
#         dlayer: bool = True,
#         chaotic_act: ChaoticAct = ChaoticAct.DRELU,
#         stable_act: nn.Module = nn.functional.mish,
#         drelu_a: float = 0.2,
#         chaotic_ratio: float = 0.2,
#     ):
#         super().__init__()

#         self.rnn = nn.LSTM(
#             input_size=input_features,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout_rate if num_layers > 1 else 0.0,
#             bidirectional=True,
#         )

#         lstm_out = hidden_size * 2

#         if dlayer:
#             self.hybrid_layer = DLayer(
#                 features=lstm_out,
#                 chaotic_act=chaotic_act,
#                 stable_act=stable_act,
#                 drelu_a=drelu_a,
#                 chaotic_ratio=chaotic_ratio,
#             )
#         else:
#             self.hybrid_layer = nn.Identity()

#         self.bn = nn.BatchNorm1d(lstm_out)
#         self.out = nn.Linear(lstm_out, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 2:
#             x = x.unsqueeze(1)

#         out, _ = self.rnn(x)
#         last = out[:, -1, :]
#         # last = self.bn(last)
#         d_out = self.hybrid_layer(last)
#         return F.softplus(self.out(d_out))


# # ── Usage ─────────────────────────────────────────────────────────────────────

# # if __name__ == "__main__":
# #     x = torch.randn(32, 15)

# #     # pick one chaotic activation
# #     m_heaviside = ForwardModel(dlayer=True, chaotic_act=ChaoticAct.HEAVISIDE)
# #     m_drelu = ForwardModel(dlayer=True, chaotic_act=ChaoticAct.DRELU)
# #     m_dexp = ForwardModel(dlayer=True, chaotic_act=ChaoticAct.DEXP)
# #     m_off = ForwardModel(dlayer=False)

# #     for name, m in [
# #         ("Heaviside", m_heaviside),
# #         ("D-ReLU", m_drelu),
# #         ("D-Exp", m_dexp),
# #         ("No D-layer", m_off),
# #     ]:
# #         n = sum(p.numel() for p in m.parameters())
# #         print(f"{name:12s}  out={tuple(m(x).shape)}  params={n:,}")

import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(
        self,
        input_features=15,
        hidden_size=200,
        dropout_rate=0.0,
        num_layers=5,
        dlayer=True,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.dlayer = dlayer
        if self.dlayer:
            self.hybrid_layer = HybridDLayer(hidden_size * 2)
        self.out = nn.Linear(hidden_size * 2, 1)
        self.bn = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        # last_out = self.bn(last_out)
        # two hybrid layers
        # last_out = self.ln(last_out)4
        if self.dlayer:
            last_out = self.hybrid_layer(last_out)
        # h_out = self.hybrid_layer(d_out)

        return nn.functional.softplus(self.out(last_out))


# Custom Autograd function for Heaviside/Chaotic gradients as seen in [8]
class ChaoticBase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, type="heaviside", a=0.2):
        if type == "heaviside":
            return (x >= 0).float()
        elif type == "d_relu":
            return torch.where(x >= a, x, torch.zeros_like(x))
        elif type == "d_exponential":
            # clamped = torch.clamp(torch.exp(x), min=1.0, max=3.0)
            return torch.where(x >= 0, torch.exp(x), torch.zeros_like(x))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Sources suggest setting chaotic gradients to zero for stability [8]
        return grad_output * 0, None, None


class HybridDLayer(nn.Module):
    def __init__(self, units, chaotic_ratio=0.2):
        super().__init__()
        self.units = units
        # Determine how many neurons will be chaotic (D-type)
        self.n_chaotic = int(units * chaotic_ratio)
        self.n_stable = units - self.n_chaotic

        # Standard linear transformations
        self.stable_linear = nn.Linear(units, units)
        self.chaotic_linear = nn.Linear(self.n_chaotic, self.n_chaotic)

        # Learnable parameters for D-neurons as defined in the sources:
        # mu (constant shift) and alpha (jump scalar/weight for chaos)
        self.mu = nn.Parameter(torch.zeros(self.n_chaotic))
        self.alpha = nn.Parameter(torch.ones(self.n_chaotic))
        self.pre_ln = nn.LayerNorm(units)

    def forward(self, x):
        # 1. Compute stable activation (f^c) for all neurons
        # Sources often use ReLU or Sigmoid as the stable base [5, 6]
        base_out = self.stable_linear(x)
        base_out = self.pre_ln(base_out)
        base_out = torch.nn.functional.mish(base_out)

        # 2. Split neurons into C-type and D-type
        stable_part = base_out[:, : self.n_stable]
        chaotic_target_part = base_out[:, self.n_stable :]

        # 3. Compute chaotic component (f^d) for D-type neurons only
        # Example using D-Exponential (D3): exp(x) if x >= 0 else 0 [7]
        # raw_chaos = torch.where(chaotic_target_part >= 0,
        #                         torch.exp(chaotic_target_part),
        #                         torch.zeros_like(chaotic_target_part))
        chaos_in = self.chaotic_linear(chaotic_target_part)
        raw_chaos = ChaoticBase.apply(chaos_in, "d_exponential")

        # Apply weighting (alpha) and shift (mu) to the chaotic component [8]
        # Weighted Chaos: alpha * f^d
        weighted_chaos = self.alpha * raw_chaos

        # 4. Aggregate D-neurons: mu + f^c + weighted_chaos [8]
        d_neurons = self.mu + chaos_in + weighted_chaos

        # 5. Concatenate C-neurons and D-neurons back together
        return torch.cat([stable_part, d_neurons], dim=1)


# class DLayer(nn.Module):
#     def __init__(self, units):
#         super().__init__()
#         # Linear transformations for stable and chaotic components [10, 12]
#         self.stable_linear = nn.Linear(units, units)
#         self.chaotic_linear = nn.Linear(units, units)
#         self.shift = nn.Parameter(torch.zeros(units))  # Constant shift mu [9]

#     def forward(self, x):
#         # f^c: Stable activation (e.g., ReLU) [10, 13]
#         stable_out = torch.nn.functional.mish(self.stable_linear(x))

#         # f^d: Chaotic activation (e.g., D-Exponential) [10, 11]
#         chaotic_val = ChaoticBase.apply(x, "d_exponential")
#         chaotic_out = self.chaotic_linear(chaotic_val)

#         # Aggregation of components: mu + f^c + f^d [9, 10]
#         return self.shift + stable_out + chaotic_out
