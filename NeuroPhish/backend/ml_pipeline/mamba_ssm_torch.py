
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union, Tuple, Optional

@dataclass
class MambaConfig:
    d_model: int = 768  # Matches Wav2Vec2 hidden size
    n_layer: int = 4    # Number of Mamba layers
    d_state: int = 16   # SSM state expansion factor
    expand: int = 2     # Block expansion factor
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4     # Local convolution width
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Projects input to hidden state
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        
        # Convolution processing
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        
        # SSM Parameters
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + config.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.act = nn.SiLU()

    def ssm(self, x: torch.Tensor):
        """
        Runs the SSM (Selective State Space Model) scan.
        x: (Batch, SeqLen, InnerDim)
        """
        (b, l, d) = x.shape
        
        # 1. Project to parameters (Delta, B, C)
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        
        (delta, B, C) = x_dbl.split(
            [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1
        )
        
        # 2. Parameterize Delta
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d)
        
        # 3. Discretization
        # A is diagonal in Mamba (S4D)
        A = -torch.exp(self.A_log.float())  # (d, state)
        # We need to broadcast A to (b, l, d, state) for the scan
        # This is the simplified parallel scan approximation for inference
        
        # NOTE: For true Mamba efficiency we'd needs CUDA primitives (selective_scan_cuda).
        # Here we implement a PyTorch-native recurrent loop which is slower but correct for CPU/MPS.
        
        ys = []
        h = torch.zeros(b, d, self.config.d_state, device=x.device) # Hidden state
        
        # Recurrent scan
        for t in range(l):
            dt = delta[:, t, :].unsqueeze(-1)  # (b, d, 1)
            dA = torch.exp(A * dt)             # (b, d, state)
            dB = B[:, t, :].unsqueeze(1) * dt  # (b, 1, state) * (b, d, 1) -> (b, d, state)
            
            # State update: h_t = A_bar * h_{t-1} + B_bar * x_t
            xt = x[:, t, :].unsqueeze(-1)      # (b, d, 1)
            h = dA * h + dB * xt               # (b, d, state)
            
            # Output: y_t = C_t * h_t
            Ct = C[:, t, :].unsqueeze(1)       # (b, 1, state)
            y = torch.sum(h * Ct, dim=-1)      # (b, d)
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (b, l, d)
        
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, D_Model)
        (b, l, d) = x.shape
        
        # 1. Expand
        x_and_res = self.in_proj(x)  # (b, l, 2*inner)
        (x, res) = x_and_res.split(self.config.d_inner, dim=-1)
        
        # 2. Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        
        x = self.act(x)
        
        # 3. SSM
        y = self.ssm(x)
        
        # 4. Gating
        y = y * self.act(res)
        
        # 5. Output
        return self.out_proj(y)

class Detect2B(nn.Module):
    """
    Full DETECT-2B Architecture:
    Wav2Vec2 Embeddings -> Adapter -> Mamba Layers -> Classifier Head
    """
    def __init__(self, config: MambaConfig = None):
        super().__init__()
        if config is None:
            config = MambaConfig()
        
        self.config = config
        
        # Mamba Backbone
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layer)
        ])
        
        self.norm_f = nn.LayerNorm(config.d_model)
        
        # Classifier Head (Sequence Modeling -> Token Classification)
        # We output a score PER FRAME (SeqLen)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1), # Binary classification (Real vs Fake)
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, D_Model) - from Wav2Vec2
        
        for layer in self.layers:
            x = x + layer(x) # Residual connection
            
        x = self.norm_f(x)
        
        # Frame-level predictions
        logits = self.head(x) # (Batch, SeqLen, 1)
        return logits
