import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
TRAIN_RESULTS_DIR = "./training_reports"    # Check match with train.py
INFERENCE_RESULTS_DIR = "./inference_reports" # Check match with inference.py
OUTPUT_DIR = "./analysis_output"              

# Define the order for charts
IMPL_ORDER = ["reference", "cuda", "cuda_opt", "cuda_opt2"]

def load_data_robust(search_path, type_label):
    """Generic function to load JSONs robustly."""
    data = []
    files = glob.glob(search_path)
    print(f"   📂 [{type_label}] Searching: {search_path}")
    print(f"      Found {len(files)} files.")
    
    for f in files:
        with open(f, 'r') as file:
            try:
                content = json.load(file)
                # Handle lists (inference) vs dicts (training)
                if isinstance(content, list):
                    data.extend(content)
                else:
                    data.append(content)
            except Exception as e:
                print(f"      ⚠️ Error reading {f}: {e}")
    return pd.DataFrame(data)

def get_training_df():
    df = load_data_robust(os.path.join(TRAIN_RESULTS_DIR, "*.json"), "TRAIN")
    if df.empty: return df
    
    # Normalize column names if needed
    col_map = {
        'throughput_img_sec': 'train_throughput',
        'peak_vram_mb': 'vram_mb',
        'avg_fwd_ms': 'fwd_ms',
        'avg_bwd_ms': 'bwd_ms',
        'total_time_s': 'total_train_time'
    }
    return df.rename(columns=col_map)

def get_inference_df():
    df = load_data_robust(os.path.join(INFERENCE_RESULTS_DIR, "*.json"), "INFER")
    if df.empty: return df
    
    col_map = {
        'throughput_imgs_sec': 'inf_throughput',
        'latency_ms': 'inf_latency',
        'accuracy_percent': 'accuracy'
    }
    return df.rename(columns=col_map)

def plot_speedup(df):
    """Bar chart with Error Bars showing speedup vs Reference."""
    if df.empty or 'inf_throughput' not in df.columns: return

    plt.figure(figsize=(10, 6))
    
    # 1. Calculate Baseline Mean (Reference)
    ref_df = df[df['implementation'] == 'reference']
    if ref_df.empty:
        print("      ⚠️ Reference missing; cannot calculate relative speedup.")
        return
    baseline_throughput = ref_df['inf_throughput'].mean()
    
    # 2. Normalize all runs against that baseline
    df = df.copy()
    df['speedup'] = df['inf_throughput'] / baseline_throughput
    
    # 3. Plot (Seaborn automatically calculates mean & confidence interval)
    sns.barplot(
        data=df, x='implementation', y='speedup', 
        palette='viridis', order=IMPL_ORDER, capsize=0.1, errorbar='sd'
    )
    
    plt.title('Inference Speedup (Mean ± Std Dev)', fontsize=14)
    plt.ylabel('Speedup Factor (x Times Faster)')
    plt.xlabel('Implementation')
    plt.axhline(1, color='red', linestyle='--', label='Baseline')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_inference_speedup.png"))
    print("   📊 Saved Speedup Chart")

def plot_breakdown(df):
    """Stacked bar chart for Forward vs Backward pass (Mean values)."""
    if df.empty or 'fwd_ms' not in df.columns: return

    # Average across seeds for the stacked plot
    df_avg = df.groupby('implementation')[['fwd_ms', 'bwd_ms']].mean()
    
    # Reorder if indices exist
    existing_order = [x for x in IMPL_ORDER if x in df_avg.index]
    df_avg = df_avg.reindex(existing_order)
    
    if df_avg.empty: return

    ax = df_avg.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
    
    plt.title('Training Kernel Breakdown (Average)', fontsize=14)
    plt.ylabel('Time per Batch (ms)')
    plt.xlabel('Implementation')
    plt.legend(["Forward Pass", "Backward Pass"])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_kernel_breakdown.png"))
    print("   📊 Saved Kernel Breakdown Chart")

# def plot_tradeoff(df):
#     """Scatter plot showing all seed runs."""
#     if df.empty or 'inf_latency' not in df.columns: return

#     plt.figure(figsize=(8, 6))
    
#     # Plot all individual run points
#     sns.scatterplot(
#         data=df, x='inf_latency', y='accuracy', hue='implementation', 
#         style='implementation', s=100, palette='deep', alpha=0.8
#     )
    
#     # Optionally plot the Means as larger markers
#     df_mean = df.groupby('implementation')[['inf_latency', 'accuracy']].mean().reset_index()
#     sns.scatterplot(
#         data=df_mean, x='inf_latency', y='accuracy', hue='implementation', 
#         marker='X', s=300, palette='deep', legend=False, edgecolor='black'
#     )
    
#     plt.title('Trade-off: Accuracy vs. Latency (X = Mean)', fontsize=14)
#     plt.xlabel('Latency (ms) [Lower is Better] ->')
#     plt.ylabel('Accuracy (%) [Higher is Better] ->')
#     plt.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "3_accuracy_vs_latency.png"))
#     print("   📊 Saved Trade-off Chart")

def plot_tradeoff(df):
    """Scatter plot showing ONLY the mean Accuracy vs Latency."""
    if df.empty or 'inf_latency' not in df.columns: return

    plt.figure(figsize=(8, 6))
    
    # 1. Calculate Mean per Implementation
    df_mean = df.groupby('implementation')[['inf_latency', 'accuracy']].mean().reset_index()
    
    # 2. Plot ONLY the Means
    # We use 'style' to give them distinct shapes (circle, X, square, etc.)
    sns.scatterplot(
        data=df_mean, x='inf_latency', y='accuracy', hue='implementation', 
        style='implementation', s=300, palette='viridis', edgecolor='black'
    )
    
    plt.title('Trade-off: Accuracy vs. Latency (Average)', fontsize=14)
    plt.xlabel('Latency (ms) [Lower is Better] ->')
    plt.ylabel('Accuracy (%) [Higher is Better] ->')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add text labels next to the dots so you don't always need the legend
    for i, row in df_mean.iterrows():
        plt.text(
            row['inf_latency'], 
            row['accuracy'] + 0.3, # Shift text up slightly
            row['implementation'], 
            fontsize=10, 
            fontweight='bold',
            ha='center'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_accuracy_vs_latency.png"))
    print("   📊 Saved Trade-off Chart (Means Only)")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- 📥 Loading Data ---")
    train_df = get_training_df()
    inf_df = get_inference_df()
    
    if train_df.empty and inf_df.empty:
        print("❌ No data found. Exiting.")
        return

    # --- 1. Aggregation (The "Average" Table) ---
    print("\n--- 📝 Statistical Summary (Mean ± Std) ---")
    
    # We aggregate separately first to avoid Cartesian product issues 
    # (e.g. merging 3 train runs with 3 inference runs = 9 rows)
    
    # Stats to calculate
    train_agg = pd.DataFrame()
    if not train_df.empty:
        train_agg = train_df.groupby('implementation').agg({
            'train_throughput': ['mean', 'std'],
            'vram_mb': ['mean', 'std'],
            'fwd_ms': ['mean'],
            'bwd_ms': ['mean']
        })

    inf_agg = pd.DataFrame()
    if not inf_df.empty:
        inf_agg = inf_df.groupby('implementation').agg({
            'inf_latency': ['mean', 'std'],
            'inf_throughput': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        })
    
    # Join them on index (implementation)
    full_stats = pd.concat([inf_agg, train_agg], axis=1)
    
    # Rounding for display
    print(full_stats.round(2).to_string())
    
    csv_path = os.path.join(OUTPUT_DIR, "final_stats_summary.csv")
    full_stats.to_csv(csv_path)
    print(f"\nSaved Stats to: {csv_path}")

    # --- 2. Plotting ---
    print("\n--- 🎨 Generating Plots ---")
    if not inf_df.empty:
        plot_speedup(inf_df)  # Use raw data for error bars
        plot_tradeoff(inf_df) # Use raw data for scatter
        
    if not train_df.empty:
        plot_breakdown(train_df) # Uses mean internally

    print(f"\n✅ Analysis Complete. Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()