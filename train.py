# import argparse
# import time
# import sys
# import os
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.models import resnet18, resnet50
# from datetime import datetime

# # Import your surgery tool
# from model_surgery import replace_conv_with_wtconv

# # --- 1. Argument Parsing ---
# def get_args():
#     parser = argparse.ArgumentParser(description="WTConv Evaluation Suite")
    
#     # Experiment Config
#     parser.add_argument('--impl', type=str, required=True, choices=['baseline', 'reference', 'cpp', 'cuda', "cuda_opt", "cuda_opt2"],
#                         help="Which implementation to test.")
#     parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'],
#                         help="Model architecture to use.")
    
#     # Hyperparameters
#     parser.add_argument('--batch-size', type=int, default=64, help="Input batch size.")
#     parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs.")
#     parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    
#     # System Config
#     parser.add_argument('--device', type=str, default='cuda', help="Device to run on.")
#     parser.add_argument('--num-workers', type=int, default=4, help="Data loader workers.")
#     parser.add_argument('--dry-run', action='store_true', help="Run only 10 batches to test.")
    
#     # Output Config
#     parser.add_argument('--results-dir', type=str, default='./experiment_results', 
#                         help="Directory to store JSON results and checkpoints.")
#     parser.add_argument('--save-weights', action='store_true', help="Save model checkpoint.")

#     return parser.parse_args()

# # --- 2. Data Factory ---
# def get_dataset(batch_size, num_workers):
#     print(f"--- 📂 Loading Imagenette (Batch={batch_size}) ---")
#     stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(*stats)
#     ])
    
#     # Ensure data folder exists
#     root_dir = os.path.join(os.path.dirname(__file__), 'data')
#     os.makedirs(root_dir, exist_ok=True)

#     try:
#         trainset = torchvision.datasets.Imagenette(root=root_dir, split='train', download=True, transform=transform)
#     except RuntimeError as e:
#         if "already exists" in str(e):
#             print("    ⚠️  Dataset directory exists. Attempting to load without download...")
#             trainset = torchvision.datasets.Imagenette(root=root_dir, split='train', download=False, transform=transform)
#         else:
#             raise e

#     loader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
#     )
#     return loader

# # --- 3. Model Factory ---
# def get_model(arch, impl_type, device):
#     print(f"--- 🏗️  Building Model: {arch.upper()} [{impl_type}] ---")
    
#     if arch == 'resnet18':
#         model = resnet18(num_classes=10)
#     elif arch == 'resnet50':
#         model = resnet50(num_classes=10)
#     else:
#         raise ValueError(f"Unknown architecture: {arch}")

#     if impl_type != 'baseline':
#         print(f"    Performing surgery to inject '{impl_type}' layers...")
#         model = replace_conv_with_wtconv(model, target_impl=impl_type, verbose=True)
#     else:
#         print("    Using standard nn.Conv2d layers.")

#     return model.to(device)

# # # --- 4. Training Engine ---
# # def train_model(args, model, loader):
# #     criterion = nn.CrossEntropyLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
# #     inference_only = (args.impl == 'cpp')
# #     if inference_only:
# #         print("⚠️  WARNING: Running in INFERENCE ONLY mode (No Backprop).")

# #     start_event = torch.cuda.Event(enable_timing=True)
# #     end_event = torch.cuda.Event(enable_timing=True)
    
# #     model.train()
# #     torch.cuda.reset_peak_memory_stats()
    
# #     print(f"\n--- 🚀 Starting Training ({args.epochs} epochs) ---")
    
# #     total_samples = 0
# #     total_compute_time = 0.0
    
# #     for epoch in range(args.epochs):
# #         for i, (inputs, labels) in enumerate(loader):
# #             if args.dry_run and i >= 10: break

# #             inputs, labels = inputs.to(args.device), labels.to(args.device)
            
# #             start_event.record()
# #             optimizer.zero_grad()
# #             outputs = model(inputs)
# #             loss = criterion(outputs, labels)
            
# #             if not inference_only:
# #                 loss.backward()
# #                 optimizer.step()
            
# #             end_event.record()
# #             torch.cuda.synchronize()
            
# #             batch_ms = start_event.elapsed_time(end_event)
# #             total_compute_time += (batch_ms / 1000.0)
# #             total_samples += inputs.size(0)
            
# #             if i % 20 == 0:
# #                 print(f"    Epoch [{epoch+1}/{args.epochs}] Step [{i}] Loss: {loss.item():.4f} | Batch Time: {batch_ms:.2f}ms")

# #     peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
# #     throughput = total_samples / total_compute_time if total_compute_time > 0 else 0
    
# #     return {
# #         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #         "implementation": args.impl,
# #         "model": args.model,
# #         "batch_size": args.batch_size,
# #         "epochs": args.epochs,
# #         "throughput_img_sec": round(throughput, 2),
# #         "peak_vram_mb": round(peak_mem, 2),
# #         "total_time_s": round(total_compute_time, 2),
# #         "device": args.device
# #         }

# def train_model(args, model, loader):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
#     inference_only = (args.impl == 'cpp')
#     if inference_only:
#         print("⚠️  WARNING: Running in INFERENCE ONLY mode (No Backprop).")

#     # --- Setup Timing Events ---
#     # We need separate events for Forward and Backward to see where the bottleneck is
#     start_batch = torch.cuda.Event(enable_timing=True)
#     end_batch   = torch.cuda.Event(enable_timing=True)
    
#     start_fwd   = torch.cuda.Event(enable_timing=True)
#     end_fwd     = torch.cuda.Event(enable_timing=True)
    
#     start_bwd   = torch.cuda.Event(enable_timing=True)
#     end_bwd     = torch.cuda.Event(enable_timing=True)
    
#     model.train()
#     torch.cuda.reset_peak_memory_stats()
    
#     print(f"\n--- 🚀 Starting Training ({args.epochs} epochs) ---")
    
#     total_samples = 0
#     total_batch_time = 0.0
#     total_fwd_time = 0.0
#     total_bwd_time = 0.0
    
#     # Track loss to ensure the model isn't outputting garbage (NaNs/Zeros)
#     last_loss = 0.0 
    
#     for epoch in range(args.epochs):
#         for i, (inputs, labels) in enumerate(loader):
#             if args.dry_run and i >= 10: break

#             inputs, labels = inputs.to(args.device), labels.to(args.device)
#             optimizer.zero_grad()
            
#             # 1. Measure Total Batch Time
#             start_batch.record()

#             # 2. Measure Forward
#             start_fwd.record()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             end_fwd.record()
            
#             # 3. Measure Backward (if applicable)
#             if not inference_only:
#                 start_bwd.record()
#                 loss.backward()
#                 optimizer.step()
#                 end_bwd.record()
            
#             end_batch.record()
            
#             # Wait for GPU to finish everything
#             torch.cuda.synchronize()
            
#             # Accumulate Times
#             batch_ms = start_batch.elapsed_time(end_batch)
#             fwd_ms   = start_fwd.elapsed_time(end_fwd)
            
#             # Handle inference case where bwd time is 0
#             bwd_ms = 0.0
#             if not inference_only:
#                 bwd_ms = start_bwd.elapsed_time(end_bwd)

#             total_batch_time += (batch_ms / 1000.0) # convert to seconds
#             total_fwd_time   += (fwd_ms / 1000.0)
#             total_bwd_time   += (bwd_ms / 1000.0)
            
#             total_samples += inputs.size(0)
#             last_loss = loss.item()
            
#             if i % 20 == 0:
#                 print(f"    Epoch [{epoch+1}/{args.epochs}] Step [{i}] "
#                       f"Loss: {last_loss:.4f} | "
#                       f"Fwd: {fwd_ms:.1f}ms | Bwd: {bwd_ms:.1f}ms")

#     # --- Final Calculations ---
#     peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
#     throughput = total_samples / total_batch_time if total_batch_time > 0 else 0
    
#     # Avoid division by zero if dry_run was too short
#     num_batches = (i + 1) + (epoch * len(loader))
#     if num_batches == 0: num_batches = 1
    
#     avg_fwd_ms = (total_fwd_time * 1000) / num_batches
#     avg_bwd_ms = (total_bwd_time * 1000) / num_batches

#     return {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "implementation": args.impl,
#         "model": args.model,
#         "batch_size": args.batch_size,
#         "epochs": args.epochs,
#         "throughput_img_sec": round(throughput, 2),
#         "peak_vram_mb": round(peak_mem, 2),
#         "total_time_s": round(total_batch_time, 2),
#         "avg_fwd_ms": round(avg_fwd_ms, 2),
#         "avg_bwd_ms": round(avg_bwd_ms, 2),
#         "final_loss": round(last_loss, 4),
#         "device": args.device
#     }

# # --- 5. Main Driver ---
# def main():
#     args = get_args()
    
#     if args.device == 'cuda' and not torch.cuda.is_available():
#         print("Error: CUDA requested but not available.")
#         sys.exit(1)

#     # 1. Setup Directories
#     os.makedirs(args.results_dir, exist_ok=True)
    
#     # Generate Unique Run Name: e.g., "resnet18_baseline_b64_20231027_143005"
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_name = f"{args.model}_{args.impl}_b{args.batch_size}_{timestamp}"
    
#     loader = get_dataset(args.batch_size, args.num_workers)
#     model = get_model(args.model, args.impl, args.device)
    
#     try:
#         metrics = train_model(args, model, loader)
        
#         # 2. Save JSON Result
#         json_path = os.path.join(args.results_dir, f"{run_name}.json")
#         with open(json_path, 'w') as f:
#             json.dump(metrics, f, indent=4)
        
#         print("\n--- 📊 Final Metrics ---")
#         print(json.dumps(metrics, indent=4))
#         print(f"📄 Result saved to: {json_path}")
            
#         # 3. Save Weights (if requested)
#         if args.save_weights:
#             weights_path = os.path.join(args.results_dir, f"{run_name}.pth")
#             torch.save(model.state_dict(), weights_path)
#             print(f"💾 Checkpoint saved: {weights_path}")
            
#     except Exception as e:
#         print(f"\n❌ Experiment Failed: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()




import argparse
import time
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from datetime import datetime

# Import your surgery tool
from model_surgery import replace_conv_with_wtconv

# --- 1. Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="WTConv Evaluation Suite")
    
    # Experiment Config
    parser.add_argument('--impl', type=str, required=True, choices=['baseline', 'reference', 'cpp', 'cuda', "cuda_opt", "cuda_opt2"],
                        help="Which implementation to test.")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'],
                        help="Model architecture to use.")
    
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, help="Input batch size.")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    
    # System Config
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on.")
    parser.add_argument('--num-workers', type=int, default=4, help="Data loader workers.")
    parser.add_argument('--dry-run', action='store_true', help="Run only 10 batches to test.")
    
    # Output Config
    parser.add_argument('--results-dir', type=str, default='./experiment_results', 
                        help="Directory to store JSON results.")
    
    # --- NEW: Cache Config ---
    parser.add_argument('--cache-dir', type=str, default='./model_cache', 
                        help="Directory to store model checkpoints.")
    parser.add_argument('--save-weights', action='store_true', help="Save model checkpoint.")

    return parser.parse_args()

# --- 2. Data Factory ---
def get_dataset(batch_size, num_workers):
    print(f"--- 📂 Loading Imagenette (Batch={batch_size}) ---")
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    # Ensure data folder exists
    root_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(root_dir, exist_ok=True)

    try:
        trainset = torchvision.datasets.Imagenette(root=root_dir, split='train', download=True, transform=transform)
    except RuntimeError as e:
        if "already exists" in str(e):
            print("    ⚠️  Dataset directory exists. Attempting to load without download...")
            trainset = torchvision.datasets.Imagenette(root=root_dir, split='train', download=False, transform=transform)
        else:
            raise e

    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    return loader

# --- 3. Model Factory ---
def get_model(arch, impl_type, device):
    print(f"--- 🏗️  Building Model: {arch.upper()} [{impl_type}] ---")
    
    if arch == 'resnet18':
        model = resnet18(num_classes=10)
    elif arch == 'resnet50':
        model = resnet50(num_classes=10)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    if impl_type != 'baseline':
        print(f"    Performing surgery to inject '{impl_type}' layers...")
        model = replace_conv_with_wtconv(model, target_impl=impl_type, verbose=True)
    else:
        print("    Using standard nn.Conv2d layers.")

    return model.to(device)

# --- 4. Training Engine ---
def train_model(args, model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    inference_only = (args.impl == 'cpp')
    if inference_only:
        print("⚠️  WARNING: Running in INFERENCE ONLY mode (No Backprop).")

    # --- Setup Timing Events ---
    start_batch = torch.cuda.Event(enable_timing=True)
    end_batch   = torch.cuda.Event(enable_timing=True)
    
    start_fwd   = torch.cuda.Event(enable_timing=True)
    end_fwd     = torch.cuda.Event(enable_timing=True)
    
    start_bwd   = torch.cuda.Event(enable_timing=True)
    end_bwd     = torch.cuda.Event(enable_timing=True)
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n--- 🚀 Starting Training ({args.epochs} epochs) ---")
    
    total_samples = 0
    total_batch_time = 0.0
    total_fwd_time = 0.0
    total_bwd_time = 0.0
    
    # Track loss to ensure the model isn't outputting garbage (NaNs/Zeros)
    last_loss = 0.0 
    
    for epoch in range(args.epochs):
        for i, (inputs, labels) in enumerate(loader):
            if args.dry_run and i >= 10: break

            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            
            # 1. Measure Total Batch Time
            start_batch.record()

            # 2. Measure Forward
            start_fwd.record()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            end_fwd.record()
            
            # 3. Measure Backward (if applicable)
            if not inference_only:
                start_bwd.record()
                loss.backward()
                optimizer.step()
                end_bwd.record()
            
            end_batch.record()
            
            # Wait for GPU to finish everything
            torch.cuda.synchronize()
            
            # Accumulate Times
            batch_ms = start_batch.elapsed_time(end_batch)
            fwd_ms   = start_fwd.elapsed_time(end_fwd)
            
            # Handle inference case where bwd time is 0
            bwd_ms = 0.0
            if not inference_only:
                bwd_ms = start_bwd.elapsed_time(end_bwd)

            total_batch_time += (batch_ms / 1000.0) # convert to seconds
            total_fwd_time   += (fwd_ms / 1000.0)
            total_bwd_time   += (bwd_ms / 1000.0)
            
            total_samples += inputs.size(0)
            last_loss = loss.item()
            
            if i % 20 == 0:
                print(f"    Epoch [{epoch+1}/{args.epochs}] Step [{i}] "
                      f"Loss: {last_loss:.4f} | "
                      f"Fwd: {fwd_ms:.1f}ms | Bwd: {bwd_ms:.1f}ms")

    # --- Final Calculations ---
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    throughput = total_samples / total_batch_time if total_batch_time > 0 else 0
    
    # Avoid division by zero if dry_run was too short
    num_batches = (i + 1) + (epoch * len(loader))
    if num_batches == 0: num_batches = 1
    
    avg_fwd_ms = (total_fwd_time * 1000) / num_batches
    avg_bwd_ms = (total_bwd_time * 1000) / num_batches

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "implementation": args.impl,
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "throughput_img_sec": round(throughput, 2),
        "peak_vram_mb": round(peak_mem, 2),
        "total_time_s": round(total_batch_time, 2),
        "avg_fwd_ms": round(avg_fwd_ms, 2),
        "avg_bwd_ms": round(avg_bwd_ms, 2),
        "final_loss": round(last_loss, 4),
        "device": args.device
    }

# --- 5. Main Driver ---
def main():
    args = get_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Error: CUDA requested but not available.")
        sys.exit(1)

    # 1. Setup Directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True) # Ensure cache dir exists
    
    # Generate Unique Run Name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.impl}b{args.batch_size}_{timestamp}"
    
    loader = get_dataset(args.batch_size, args.num_workers)
    model = get_model(args.model, args.impl, args.device)
    
    try:
        metrics = train_model(args, model, loader)
        
        # 2. Save JSON Result (Original Location)
        json_path = os.path.join(args.results_dir, f"{run_name}.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\n--- 📊 Final Metrics ---")
        print(json.dumps(metrics, indent=4))
        print(f"📄 Result saved to: {json_path}")
            
        # 3. Save Weights (To CACHE Directory)
        if args.save_weights:
            weights_path = os.path.join(args.cache_dir, f"{run_name}.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"💾 Checkpoint saved: {weights_path}")
            
    except Exception as e:
        print(f"\n❌ Experiment Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()