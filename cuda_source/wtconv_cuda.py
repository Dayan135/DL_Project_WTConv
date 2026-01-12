import torch
import torch.nn as nn
from torch.autograd import Function
import sys
import os
import torch.nn.functional as F

# --- Path Fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import cuda_module 
except ImportError:
    cuda_module = None 

class WTConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, stride, padding, groups, dwt_scale, idwt_scale, *weights):
        if cuda_module is None:
            raise RuntimeError("C++ extension not loaded.")

        # NO MORE .cpu() calls! 
        # We pass the GPU tensors directly to C++
        
        # Ensure input is contiguous (CUDA kernels hate non-contiguous memory)
        input = input.contiguous()
        
        output, saved_tensors = cuda_module.wtconv_forward_save(
            input, list(weights), stride, padding, groups, dwt_scale, idwt_scale
        )

        ctx.stride = stride
        ctx.groups = groups
        ctx.num_levels = len(weights)
        
        ctx.save_for_backward(*weights, *saved_tensors)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if cuda_module is None:
            raise RuntimeError("C++ extension not loaded.")

        # Ensure contiguous
        grad_output = grad_output.contiguous()

        # Unpack
        num_levels = ctx.num_levels
        saved_all = ctx.saved_tensors
        weights = list(saved_all[:num_levels])
        saved_activations = list(saved_all[num_levels:])
        
        # Call CUDA Backward
        grad_input, grad_weights = cuda_module.wtconv_backward(
            saved_activations, grad_output, weights, ctx.groups
        )

        return (
            grad_input, 
            None, None, None, None, None, 
            *grad_weights
        )

class WTConv2d_CUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 bias=True, wt_levels=1, wt_type='db1', padding=None):
        super().__init__()
        
        # ... (Keep __init__ exactly as it was) ...
        self.stride = stride if isinstance(stride, int) else stride[0]
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding if isinstance(padding, int) else padding[0]
            
        self.groups = 1 
        self.wt_levels = wt_levels
        self.dwt_scale = 0.5
        self.idwt_scale = 1.0 

        self.weights = nn.ParameterList()
        for _ in range(wt_levels):
            w = nn.Parameter(torch.Tensor(
                out_channels * 4, 
                in_channels * 4, 
                kernel_size, 
                kernel_size
            ))
            nn.init.kaiming_uniform_(w, a=2.236)
            self.weights.append(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        H, W = x.shape[-2:]
        
        # --- FIX: Generic Multi-Level Padding ---
        # Input must be divisible by 2^levels to ensure valid downscaling at every step
        pad_factor = 2 ** self.wt_levels 
        
        pad_h = (pad_factor - (H % pad_factor)) % pad_factor
        pad_w = (pad_factor - (W % pad_factor)) % pad_factor
        
        x_padded = x
        if pad_h > 0 or pad_w > 0:
            # Pad strictly on the right/bottom
            # Format: (pad_left, pad_right, pad_top, pad_bottom)
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Ensure GPU transfer if needed
        if self.weights[0].is_cuda and not x_padded.is_cuda:
            x_padded = x_padded.to(self.weights[0].device)

        # Run C++ / CUDA
        out = WTConv2dFunction.apply(
            x_padded, 
            self.stride, 
            self.padding, 
            self.groups, 
            self.dwt_scale, 
            self.idwt_scale, 
            *self.weights
        )
        
        # --- FIX: Crop back to original exact size ---
        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
            
        return out