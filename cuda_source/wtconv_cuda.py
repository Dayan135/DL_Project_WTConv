import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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
        """
        weights: we may receive the SAME tensor repeated wt_levels times.
        We still save them to keep the signature stable.
        """
        if cuda_module is None:
            raise RuntimeError("C++ extension not loaded.")

        input = input.contiguous()

        output, saved_tensors = cuda_module.wtconv_forward_save(
            input, list(weights), stride, padding, groups, dwt_scale, idwt_scale
        )

        ctx.stride = stride
        ctx.groups = groups
        ctx.num_levels = len(weights)

        # Save both weights and activations
        ctx.save_for_backward(*weights, *saved_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if cuda_module is None:
            raise RuntimeError("C++ extension not loaded.")

        grad_output = grad_output.contiguous()

        num_levels = ctx.num_levels
        saved_all = ctx.saved_tensors

        weights = list(saved_all[:num_levels])
        saved_activations = list(saved_all[num_levels:])

        grad_input, grad_weights = cuda_module.wtconv_backward(
            saved_activations, grad_output, weights, ctx.groups
        )

        # If the SAME weight was passed for all levels, we must sum the per-level grads
        # and return the same summed gradient for each repeated argument position.
        if num_levels > 0:
            grad_w_sum = grad_weights[0]
            for gw in grad_weights[1:]:
                grad_w_sum = grad_w_sum + gw
            grad_weights_out = [grad_w_sum] * num_levels
        else:
            grad_weights_out = []

        return (
            grad_input,          # grad for input
            None, None, None,    # stride, padding, groups (non-tensor)
            None, None,          # dwt_scale, idwt_scale (non-tensor)
            *grad_weights_out    # one grad per *weights arg
        )


class WTConv2d_CUDA(nn.Module):
    """
    Shared-weight WTConv2d: SAME filter used across all wavelet levels.
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
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding if isinstance(padding, int) else padding[0]

        self.groups = 1
        self.wt_levels = int(wt_levels)

        self.dwt_scale = 0.5
        self.idwt_scale = 1.0

        # --- SINGLE shared weight for all levels ---
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

        # Input must be divisible by 2^levels
        pad_factor = 2 ** self.wt_levels
        pad_h = (pad_factor - (H % pad_factor)) % pad_factor
        pad_w = (pad_factor - (W % pad_factor)) % pad_factor

        x_padded = x
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Ensure device matches
        if self.weight.is_cuda and not x_padded.is_cuda:
            x_padded = x_padded.to(self.weight.device)
        elif x_padded.is_cuda and not self.weight.is_cuda:
            # If user moved input to CUDA but module isn't, follow input (common pitfall)
            self.weight.data = self.weight.data.to(x_padded.device)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(x_padded.device)

        # Pass the SAME tensor wt_levels times (signature compatible with your C++ op)
        shared_weights = [self.weight] * self.wt_levels

        out = WTConv2dFunction.apply(
            x_padded,
            self.stride,
            self.padding,
            self.groups,
            self.dwt_scale,
            self.idwt_scale,
            *shared_weights
        )

        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out
