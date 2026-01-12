# import torch
# import torch.nn as nn
# from torch.autograd import Function
# import sys
# import os

# # --- FIX START: Add current folder to path ---
# # Get the absolute path to the folder where this script (wtconv_cpp.py) lives
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add it to sys.path so Python can find 'cpp_module'
# if current_dir not in sys.path:
#     sys.path.append(current_dir)
# # --- FIX END ---
# # 1. Import your compiled C++ extension
# # (Replace 'cpp_module' with the actual name defined in your setup.py)
# try:
#     import cpp_module 
# except ImportError:
#     # Fallback for when C++ isn't compiled yet (prevents IDE errors)
#     cpp_module = None 

# class WTConv2dFunction(Function):
#     @staticmethod
#     def forward(ctx, input, stride, padding, groups, dwt_scale, idwt_scale, *weights):
#         """
#         The bridge between PyTorch and C++.
#         We accept a variable number of weight tensors (*weights) to handle 'wt_levels'.
#         """
#         if cpp_module is None:
#             raise RuntimeError("C++ extension not loaded.")

#         # 1. Call C++ Forward
#         # We assume weights is a tuple of tensors, we need to convert to list for C++ binding if needed
#         # (Pybind11 usually handles tuple->std::vector conversion automatically)
#         output, saved_tensors = cpp_module.wtconv_forward_save(
#             input, list(weights), stride, padding, groups, dwt_scale, idwt_scale
#         )

#         # 2. Save context for Backward
#         # We need to save:
#         # - The intermediate activations (saved_tensors)
#         # - The weights (for gradient calc)
#         # - Integers/Floats (ctx.config)
        
#         ctx.stride = stride
#         ctx.groups = groups
#         ctx.num_levels = len(weights)
        
#         # save_for_backward requires unpacked tensors
#         # Structure: [w0, w1, ... , saved0, saved1, ...]
#         ctx.save_for_backward(*weights, *saved_tensors)

#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         if cpp_module is None:
#             raise RuntimeError("C++ extension not loaded.")

#         # 1. Unpack saved tensors
#         num_levels = ctx.num_levels
#         saved_all = ctx.saved_tensors
        
#         weights = list(saved_all[:num_levels])
#         saved_activations = list(saved_all[num_levels:])
        
#         # 2. Call C++ Backward
#         # Returns: (grad_input, [grad_w0, grad_w1, ...])
#         grad_input, grad_weights = cpp_module.wtconv_backward(
#             saved_activations, grad_output, weights, ctx.groups
#         )

#         # 3. Return gradients matching forward signature
#         # Forward inputs were: (ctx, input, stride, padding, groups, dwt_scale, idwt_scale, *weights)
#         # We only return gradients for tensors. Return None for ints/floats.
        
#         # Structure: grad_input, None, None, None, None, None, grad_w0, grad_w1...
        
#         grad_w_tuple = tuple(grad_weights)
        
#         return (
#             grad_input,  # grad for input
#             None,        # stride
#             None,        # padding
#             None,        # groups
#             None,        # dwt_scale
#             None,        # idwt_scale
#             *grad_w_tuple # unpacked gradients for weights
#         )

# class WTConv2d_CPP(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
#                  bias=True, wt_levels=1, wt_type='db1', padding=None):
#         super().__init__()
        
#         self.stride = stride if isinstance(stride, int) else stride[0]
#         # Handle padding logic (default to same padding if not provided)
#         if padding is None:
#             self.padding = (kernel_size - 1) // 2
#         else:
#             self.padding = padding if isinstance(padding, int) else padding[0]
            
#         self.groups = 1 # Logic from your script says groups is ignored/forced
#         self.wt_levels = wt_levels
        
#         # Scales for Haar (db1)
#         self.dwt_scale = 0.5
#         self.idwt_scale = 1.0 # Adjust based on your specific math requirements

#         # --- WEIGHT INITIALIZATION ---
#         # Crucial Note: 
#         # C++ does DWT (Channels * 4) -> Conv -> IDWT.
#         # So the Convolution weights must handle 4x the channels!
        
#         self.weights = nn.ParameterList()
#         for _ in range(wt_levels):
#             # Shape: [Out*4, In*4, K, K] because DWT expands channels by 4
#             w = nn.Parameter(torch.Tensor(
#                 out_channels * 4, 
#                 in_channels * 4, 
#                 kernel_size, 
#                 kernel_size
#             ))
#             nn.init.kaiming_uniform_(w, a=2.236) # Kaiming init
#             self.weights.append(w)

#         # --- BIAS HANDLING ---
#         # The C++ code provided does NOT handle bias. We apply it manually here.
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channels))
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, x):
#         # 1. Run the Custom Autograd Function
#         # We unpack self.weights using * so they are passed as individual arguments
#         out = WTConv2dFunction.apply(
#             x, 
#             self.stride, 
#             self.padding, 
#             self.groups, 
#             self.dwt_scale, 
#             self.idwt_scale, 
#             *self.weights
#         )
        
#         # 2. Add Bias (since C++ didn't do it)
#         if self.bias is not None:
#             out = out + self.bias.view(1, -1, 1, 1)
            
#         return out

import torch
import torch.nn as nn
from torch.autograd import Function
import sys
import os

# --- Path Fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import cpp_module
except ImportError:
    cpp_module = None

class WTConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, stride, padding, groups, dwt_scale, idwt_scale, *weights):
        if cpp_module is None:
            raise RuntimeError("C++ extension not loaded.")

        # 1. Record the original device (likely CUDA)
        device = input.device
        
        # 2. Move EVERYTHING to CPU (because C++ impl is CPU-only)
        input_cpu = input.cpu()
        weights_cpu = [w.cpu() for w in weights]

        # 3. Call C++
        output_cpu, saved_tensors = cpp_module.wtconv_forward_save(
            input_cpu, weights_cpu, stride, padding, groups, dwt_scale, idwt_scale
        )

        # 4. Save for backward (Keep on CPU to save GPU memory)
        ctx.stride = stride
        ctx.groups = groups
        ctx.num_levels = len(weights)
        ctx.device = device # Remember where we need to return
        
        ctx.save_for_backward(*weights_cpu, *saved_tensors)

        # 5. Move output back to GPU
        return output_cpu.to(device)

    @staticmethod
    def backward(ctx, grad_output):
        if cpp_module is None:
            raise RuntimeError("C++ extension not loaded.")

        # 1. Move gradient to CPU
        grad_output_cpu = grad_output.cpu()

        # 2. Unpack saved tensors (Already on CPU)
        num_levels = ctx.num_levels
        saved_all = ctx.saved_tensors
        weights_cpu = list(saved_all[:num_levels])
        saved_activations = list(saved_all[num_levels:])
        
        # 3. Call C++ Backward
        grad_input_cpu, grad_weights_cpu = cpp_module.wtconv_backward(
            saved_activations, grad_output_cpu, weights_cpu, ctx.groups
        )

        # 4. Move gradients back to Original Device (GPU)
        grad_input = grad_input_cpu.to(ctx.device)
        grad_weights = [gw.to(ctx.device) for gw in grad_weights_cpu]
        
        # 5. Return
        return (
            grad_input, 
            None, None, None, None, None, # ints/floats
            *grad_weights
        )

# ... (Keep the WTConv2d_CPP class exactly as it was) ...
class WTConv2d_CPP(nn.Module):
    # (Copy the __init__ and forward from previous response)
    # The forward logic here doesn't change, the Function handles the transfer.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 bias=True, wt_levels=1, wt_type='db1', padding=None):
        super().__init__()
        
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
            # Remember: C++ DWT expands channels by 4
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
        out = WTConv2dFunction.apply(
            x, 
            self.stride, 
            self.padding, 
            self.groups, 
            self.dwt_scale, 
            self.idwt_scale, 
            *self.weights
        )
        print("out shape:", out.shape)
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
            
        return out