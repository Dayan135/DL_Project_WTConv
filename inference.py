import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from datetime import datetime
import glob

# Import your surgery tool
from model_surgery import replace_conv_with_wtconv

# --- Configuration ---
# The implementations you want to search for in filenames
IMPLS_TO_FIND = ["reference", "cuda_opt2", "cuda_opt", "cuda"] 

def get_args():
    parser = argparse.ArgumentParser(description="Final Comparative Benchmark")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    
    # input / output config
    parser.add_argument('--cache-dir', type=str, default='./model_cache', 
                        help="Directory where .pth files are located.")
    parser.add_argument('--output-dir', type=str, default='./inference_reports', 
                        help="New directory where results will be saved.")
    
    return parser.parse_args()

def get_latest_checkpoint(cache_dir, model_arch, impl):
    """
    Finds the newest .pth file containing the implementation name.
    Example: matches 'resnet18_cuda_opt2_b64_...pth'
    """
    search_pattern = os.path.join(cache_dir, f"{model_arch}_{impl}_*.pth")
    files = glob.glob(search_pattern)
    
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def get_dataset(batch_size):
    # Load Validation Set
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    root = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        ds = torchvision.datasets.Imagenette(root=root, split='val', download=True, transform=transform)
    except:
        ds = torchvision.datasets.Imagenette(root=root, split='train', download=False, transform=transform)
        
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def get_model(arch, impl, device):
    if arch == 'resnet18':
        model = resnet18(num_classes=10)
    else:
        model = resnet50(num_classes=10)
        
    if impl != 'baseline':
        model = replace_conv_with_wtconv(model, target_impl=impl, verbose=False)
    
    return model.to(device)

def run_benchmark(model, loader, device):
    model.eval()
    
    # 1. Warmup
    print("      🔥 Warming up...", end="\r")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= 10: break
            _ = model(inputs.to(device))
    torch.cuda.synchronize()

    # 2. Speed Test
    print("      ⏱️  Timing...    ", end="\r")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    total_ms = 0.0
    total_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= 50: break # Benchmark 50 batches
            inputs = inputs.to(device)
            
            start_event.record()
            _ = model(inputs)
            end_event.record()
            torch.cuda.synchronize()
            
            total_ms += start_event.elapsed_time(end_event)
            total_samples += inputs.size(0)
            num_batches += 1
            
    latency = total_ms / num_batches
    throughput = total_samples / (total_ms / 1000.0)

    # 3. Accuracy Test
    print("      🎯 Checking Acc...", end="\r")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            # Check all batches for accurate result
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print("                        ", end="\r") # Clear line
    
    return latency, throughput, acc

def main():
    args = get_args()
    
    # Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    loader = get_dataset(args.batch_size)
    
    report_data = []
    
    print(f"\n=======================================================================")
    print(f"🚀 COMPARATIVE BENCHMARK REPORT")
    print(f"   Looking in: {args.cache_dir}")
    print(f"   Saving to:  {args.output_dir}")
    print(f"=======================================================================\n")

    for impl in IMPLS_TO_FIND:
        # Find file automatically
        ckpt_path = get_latest_checkpoint(args.cache_dir, args.model, impl)
        
        if not ckpt_path:
            print(f"⚠️  Skipping {impl}: No file found matching '{args.model}_{impl}*.pth'")
            continue
            
        filename = os.path.basename(ckpt_path)
        print(f"Testing [{impl}] using {filename}...")
        
        try:
            # 1. Build & Load
            model = get_model(args.model, impl, args.device)
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(checkpoint, strict=False)
            
            # 2. Run Test
            lat, thr, acc = run_benchmark(model, loader, args.device)
            
            # 3. Store Results
            result_entry = {
                "implementation": impl,
                "checkpoint": filename,
                "latency_ms": round(lat, 2),
                "throughput_imgs_sec": round(thr, 2),
                "accuracy_percent": round(acc, 2)
            }
            report_data.append(result_entry)
            
            print(f"   ✅ Done: {lat:.2f} ms | {acc:.2f}%")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # --- Generate Report Files ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save JSON
    json_path = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
        
    # 2. Save Text Table
    txt_path = os.path.join(args.output_dir, f"benchmark_summary_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        header = f"{'Implementation':<20} | {'Latency (ms)':<15} | {'Throughput':<15} | {'Accuracy %':<10}\n"
        sep = "-" * 70 + "\n"
        f.write(header)
        f.write(sep)
        
        print("\n\n" + header + sep, end="") # Print to console too
        
        for res in report_data:
            line = f"{res['implementation']:<20} | {res['latency_ms']:<15} | {res['throughput_imgs_sec']:<15} | {res['accuracy_percent']:<10}\n"
            f.write(line)
            print(line, end="")
            
    print(f"\n\n📄 Reports saved to:\n   {json_path}\n   {txt_path}")

if __name__ == "__main__":
    main()