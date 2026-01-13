import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# --- Path Fix (Same as before) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Ensure this matches the name in your setup.py (e.g., 'cuda_fused_module')
    import optimized_cuda_module as cuda_module 
except ImportError:
    cuda_module = None
    print("[WARN] cuda_module not found. Fused WTConv will fail.")


class WTConv2dFusedFunction(Function):
    @staticmethod
    def forward(ctx, input, stride, padding, groups, dwt_scale, idwt_scale, *weights):
        """
        The input 'weights' is a variable argument list (one tensor per level).
        """
        if cuda_module is None:
            raise RuntimeError("C++ extension not loaded.")

        input = input.contiguous()

        # 1. Call Fused Forward
        # Note: We pass training=True so C++ saves the inputs for backward
        output, saved_inputs = cuda_module.wtconv_forward(
            input, 
            list(weights), 
            stride, 
            padding, 
            groups, 
            dwt_scale, 
            idwt_scale, 
            True  # training flag
        )

        # 2. Save Context
        ctx.groups = groups
        ctx.dwt_scale = dwt_scale
        ctx.idwt_scale = idwt_scale
        ctx.num_levels = len(weights)

        # Save for backward: Weights + Inputs (for rematerialization)
        ctx.save_for_backward(*weights, *saved_inputs)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if cuda_module is None:
            raise RuntimeError("C++ extension not loaded.")
            
        grad_output = grad_output.contiguous()
        
        # 1. Unpack Saved Tensors
        num_levels = ctx.num_levels
        saved_all = ctx.saved_tensors
        
        weights = list(saved_all[:num_levels])
        saved_inputs = list(saved_all[num_levels:]) # These are LL inputs from forward
        
        # 2. Call Fused Backward
        # Note: New signature requires passing scales explicitly
        grad_input, grad_weights = cuda_module.wtconv_backward(
            saved_inputs, 
            grad_output, 
            weights, 
            ctx.groups, 
            ctx.dwt_scale, 
            ctx.idwt_scale
        )

        # 3. Handle Shared Weights (Accumulate Gradients)
        # If the same weight parameter was passed multiple times, we sum the grads.
        if num_levels > 0:
            grad_w_sum = grad_weights[0]
            for gw in grad_weights[1:]:
                grad_w_sum = grad_w_sum + gw
            # Replicate the summed gradient for each argument
            grad_weights_out = [grad_w_sum] * num_levels
        else:
            grad_weights_out = []

        return (
            grad_input,          # grad_input
            None, None, None,    # stride, padding, groups (non-tensor)
            None, None,          # dwt_scale, idwt_scale (non-tensor)
            *grad_weights_out    # grads for *weights
        )


class WTConv2d_Fused(nn.Module):
    """
    Optimized Fused Implementation Wrapper.
    Uses 'wtconv_forward' and 'wtconv_backward' from C++.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type="db1",
        padding=None,
    ):
        super().__init__()

        self.stride = stride if isinstance(stride, int) else stride[0]
        # Standard Conv Padding (to keep feature map size constant)
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding if isinstance(padding, int) else padding[0]

        self.groups = 1
        self.wt_levels = int(wt_levels)
        
        # Scales matching your C++ defaults
        self.dwt_scale = 0.5
        self.idwt_scale = 1.0

        # Shared Weight Parameter
        self.weight = nn.Parameter(torch.empty(
            out_channels * 4,
            in_channels * 4,
            kernel_size,
            kernel_size
        ))
        nn.init.kaiming_uniform_(self.weight, a=2.236)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        H, W = x.shape[-2:]

        # --- 1. Divisibility Padding (Safety) ---
        pad_factor = 2 ** self.wt_levels
        pad_h = (pad_factor - (H % pad_factor)) % pad_factor
        pad_w = (pad_factor - (W % pad_factor)) % pad_factor

        x_padded = x
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Handle device mismatch (User safety)
        if self.weight.is_cuda and not x_padded.is_cuda:
            x_padded = x_padded.to(self.weight.device)

        # --- 2. Run Optimized Kernel ---
        shared_weights = [self.weight] * self.wt_levels

        out = WTConv2dFusedFunction.apply(
            x_padded,
            self.stride,
            self.padding,
            self.groups,
            self.dwt_scale,
            self.idwt_scale,
            *shared_weights
        )

        # --- 3. Crop & Bias ---
        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out