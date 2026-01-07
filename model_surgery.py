import torch
import torch.nn as nn
import sys
import os

# Add local paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Reference'))

try:
    from wtconv.wtconv2d import WTConv2d
except ImportError:
    print("[WARN] Could not import WTConv2d. Surgery will fail if requested.")

def replace_conv_with_wtconv(module, container_name='model', target_impl='reference', verbose=True):
    """
    Recursively replaces nn.Conv2d with WTConv2d, BUT only if compatible.
    
    COMPATIBILITY RULES for WTConv2d (Reference):
    1. Kernel size must be > 1 (No 1x1 convs).
    2. in_channels MUST EQUAL out_channels (No dimension changes).
    3. Groups logic is ignored (WTConv forces depthwise).
    """
    
    for name, child in module.named_children():
        
        if isinstance(child, nn.Conv2d):
            
            # --- RULE 1: Skip 1x1 Convolutions ---
            if child.kernel_size == (1, 1) or child.kernel_size == 1:
                continue

            # --- RULE 2: Skip Channel Changes (The Assertion Fix) ---
            if child.in_channels != child.out_channels:
                if verbose:
                    print(f"   [SKIP] {name}: In({child.in_channels}) != Out({child.out_channels})")
                continue

            # --- PREPARE CONFIG ---
            # We explicitly only grab what WTConv2d supports
            config = {
                'in_channels': child.in_channels,
                'out_channels': child.out_channels,
                'kernel_size': child.kernel_size if isinstance(child.kernel_size, int) else child.kernel_size[0],
                'stride': child.stride if isinstance(child.stride, int) else child.stride[0],
                'bias': (child.bias is not None),
                'wt_levels': 1, 
                'wt_type': 'db1' 
            }

            # --- SWAP ---
            try:
                if target_impl == 'reference':
                    new_layer = WTConv2d(**config)
                
                # (Add your cpp/cuda placeholders here later)
                
                else:
                    raise ValueError(f"Unknown impl: {target_impl}")

                setattr(module, name, new_layer)
                if verbose:
                    print(f"✅ [SWAP] {name}: Conv2d -> WTConv (k={config['kernel_size']})")
            
            except Exception as e:
                print(f"❌ [FAIL] Could not swap {name}: {e}")

        else:
            # Recursively go deeper
            replace_conv_with_wtconv(child, f"{container_name}.{name}", target_impl, verbose)

    return module