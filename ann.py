import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(self, input_features=15, hidden_size=200, dropout_rate=0.0, num_layers=6, dlayer=True):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.d_layer = DLayer(hidden_size*2)
        self.hybrid_layer = HybridDLayer(hidden_size*2)
        self.out = nn.Linear(hidden_size*2, 1)
        self.dlayer = dlayer

    def forward(self,x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        d_out = last_out
        if self.dlayer:
            d_out = self.hybrid_layer(last_out)

        return nn.functional.softplus(self.out(d_out))



# Custom Autograd function for Heaviside/Chaotic gradients as seen in [8]
class ChaoticBase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, type='heaviside', a=0.2):
        if type == 'heaviside':
            return (x >= 0).float()
        elif type == 'd_relu':
            return torch.where(x >= a, x, torch.zeros_like(x))
        elif type == 'd_exponential':
            return torch.where(x >= 0, torch.exp(x), torch.zeros_like(x))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Sources suggest setting chaotic gradients to zero for stability [8]
        return grad_output * 0, None, None

class DLayer(nn.Module):
    def __init__(self, units):
        super().__init__()
        # Linear transformations for stable and chaotic components [10, 12]
        self.stable_linear = nn.Linear(units, units)
        self.chaotic_linear = nn.Linear(units, units)
        self.shift = nn.Parameter(torch.zeros(units)) # Constant shift mu [9]

    def forward(self, x):
        # f^c: Stable activation (e.g., ReLU) [10, 13]
        stable_out = torch.nn.functional.mish(self.stable_linear(x))

        # f^d: Chaotic activation (e.g., D-Exponential) [10, 11]
        chaotic_val = ChaoticBase.apply(x, 'd_relu')
        chaotic_out = self.chaotic_linear(chaotic_val)

        # Aggregation of components: mu + f^c + f^d [9, 10]
        return self.shift + stable_out + chaotic_out

class HybridDLayer(nn.Module):
    def __init__(self, units, chaotic_ratio=0.2):
        super().__init__()
        self.units = units
        # Determine how many neurons will be chaotic (D-type)
        self.n_chaotic = int(units * chaotic_ratio)
        self.n_stable = units - self.n_chaotic

        # Standard linear transformations
        self.stable_linear = nn.Linear(units, units)
        self.chaotic_linear = nn.Linear(units, self.n_chaotic)

        # Learnable parameters for D-neurons as defined in the sources:
        # mu (constant shift) and alpha (jump scalar/weight for chaos)
        self.mu = nn.Parameter(torch.zeros(self.n_chaotic))
        self.alpha = nn.Parameter(torch.ones(self.n_chaotic))

    def forward(self, x):
        # 1. Compute stable activation (f^c) for all neurons
        # Sources often use ReLU or Sigmoid as the stable base [5, 6]
        base_out = torch.nn.functional.mish(self.stable_linear(x))

        # 2. Split neurons into C-type and D-type
        stable_part = base_out[:, :self.n_stable]
        chaotic_target_part = base_out[:, self.n_stable:]

        # 3. Compute chaotic component (f^d) for D-type neurons only
        # Example using D-Exponential (D3): exp(x) if x >= 0 else 0 [7]
        # raw_chaos = torch.where(chaotic_target_part >= 0,
        #                         torch.exp(chaotic_target_part),
        #                         torch.zeros_like(chaotic_target_part))
        raw_chaos = ChaoticBase.apply(chaotic_target_part, 'd_exponential')

        # Apply weighting (alpha) and shift (mu) to the chaotic component [8]
        # Weighted Chaos: alpha * f^d
        weighted_chaos = self.alpha * raw_chaos

        # 4. Aggregate D-neurons: mu + f^c + weighted_chaos [8]
        d_neurons = self.mu + chaotic_target_part + weighted_chaos

        # 5. Concatenate C-neurons and D-neurons back together
        return torch.cat([stable_part, d_neurons], dim=1)
