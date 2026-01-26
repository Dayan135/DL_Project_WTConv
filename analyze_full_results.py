import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
TRAIN_RESULTS_DIR = "./training_reports"     # Folder with training JSONs
INFERENCE_RESULTS_FILE = "./inference_reports" # Folder with inference JSON(s)
OUTPUT_DIR = "./full_analysis_output"
IMPL_ORDER = ["reference", "cuda", "cuda_opt", "cuda_opt2"]
LABELS = {
    "reference": "Reference (PyTorch)",
    "cuda": "CUDA (Naive)",
    "cuda_opt": "CUDA (Shared Mem?)",
    "cuda_opt2": "CUDA (Reg Tiling?)"
}

def load_data():
    """Loads and merges data from both sources."""
    data_records = []

    # 1. Load Training Data (Individual JSONs)
    train_files = glob.glob(os.path.join(TRAIN_RESULTS_DIR, "*.json"))
    print(f"   📂 Found {len(train_files)} training files.")
    
    for f in train_files:
        try:
            with open(f, 'r') as file:
                d = json.load(file)
                data_records.append({
                    "type": "train",
                    "implementation": d.get('implementation'),
                    "wt_levels": d.get('wy_levels', d.get('wt_levels')), # Handle typo
                    "seed": d.get('seed', 0), # Default to 0 if missing
                    "train_throughput": d.get('throughput_img_sec'),
                    "fwd_ms": d.get('avg_fwd_ms'),
                    "bwd_ms": d.get('avg_bwd_ms'),
                    "vram_mb": d.get('peak_vram_mb')
                })
        except Exception as e:
            print(f"     ⚠️ Error reading {f}: {e}")

    # 2. Load Inference Data (List of JSONs)
    inf_files = glob.glob(os.path.join(INFERENCE_RESULTS_FILE, "*.json"))
    print(f"   📂 Found {len(inf_files)} inference files.")
    
    for f in inf_files:
        try:
            with open(f, 'r') as file:
                content = json.load(file)
                if isinstance(content, list):
                    for item in content:
                        data_records.append({
                            "type": "inference",
                            "implementation": item.get('implementation'),
                            "wt_levels": item.get('wt_levels'),
                            "inf_latency": item.get('latency_ms'),
                            "inf_throughput": item.get('throughput_imgs_sec'),
                            "accuracy": item.get('accuracy_percent')
                        })
        except Exception as e:
            print(f"     ⚠️ Error reading {f}: {e}")

    return pd.DataFrame(data_records)

# def plot_scalability_latency(df):
#     """Line Chart: Latency vs. Levels (Handles Overlaps)"""
#     df_inf = df[df['type'] == 'inference']
#     if df_inf.empty: return

#     plt.figure(figsize=(10, 6))
    
#     # Define distinct styles to handle perfect overlaps
#     # solid, dashed, dotted, dashdot
#     styles = ['-', '--', '-.', ':'] 
#     markers = ['o', 'X', 's', 'D']  # Circle, X, Square, Diamond
    
#     # Create the plot with manual style mapping ensures visibility
#     sns.lineplot(
#         data=df_inf, 
#         x='wt_levels', 
#         y='inf_latency', 
#         hue='implementation', 
#         style='implementation',   # This activates different line styles
#         markers=True,
#         dashes=True,              # Enable dash patterns
#         linewidth=2.5,
#         palette='viridis', 
#         hue_order=IMPL_ORDER,
#         err_style='bars',
#         errorbar='sd',
#         markersize=9
#     )
    
#     plt.title('Scalability: Inference Latency vs. Wavelet Levels', fontsize=14)
#     plt.xlabel('Wavelet Levels (Complexity)')
#     plt.ylabel('Latency (ms) [Lower is Better]')
#     plt.xticks([1, 2, 3, 4])
#     plt.grid(True, linestyle='--', alpha=0.4)
    
#     # Move legend outside if it crowds the plot
#     plt.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "1_scalability_latency.png"))
#     print("   📊 Saved Scalability Chart (With Overlap Fix)")

def plot_scalability_latency(df):
    """Grouped Bar Chart: Latency vs. Levels (Solves Overlap)"""
    df_inf = df[df['type'] == 'inference']
    if df_inf.empty: return

    plt.figure(figsize=(12, 6))
    
    # Using a Bar Plot automatically groups them side-by-side
    ax = sns.barplot(
        data=df_inf, 
        x='wt_levels', 
        y='inf_latency', 
        hue='implementation',
        palette='viridis', 
        hue_order=IMPL_ORDER,
        edgecolor='black',
        errorbar='sd',       # correct modern syntax
        err_kws={'color': 'black'}, # <--- THE FIX: Use err_kws instead of err_color
        capsize=0.1
    )
    
    plt.title('Scalability: Inference Latency vs. Wavelet Levels', fontsize=15)
    plt.xlabel('Wavelet Levels (Complexity)', fontsize=12)
    plt.ylabel('Latency (ms) [Lower is Better]', fontsize=12)
    
    # Add grid lines behind the bars
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legend outside to keep chart clean
    plt.legend(title='Implementation', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_scalability_latency_bar.png"))
    print("   📊 Saved Scalability Bar Chart")

    
def plot_throughput_comparison(df):
    """Grouped Bar Chart: Speed comparison at each level."""
    df_inf = df[df['type'] == 'inference']
    if df_inf.empty: return

    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=df_inf, x='wt_levels', y='inf_throughput', hue='implementation',
        palette='viridis', hue_order=IMPL_ORDER, errorbar='sd', capsize=0.1
    )
    
    plt.title('Throughput Comparison by Level', fontsize=14)
    plt.xlabel('Wavelet Levels')
    plt.ylabel('Images / Sec [Higher is Better]')
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_throughput_levels.png"))
    print("   📊 Saved Throughput Bar Chart")

def plot_kernel_breakdown_level4(df):
    """Stacked Bar: Where does the time go? (Only for Level 4)."""
    # Filter for Training data at Level 4
    df_train = df[(df['type'] == 'train') & (df['wt_levels'] == 4)]
    
    if df_train.empty:
        print("   ⚠️ No training data found for Level 4. Skipping breakdown.")
        return

    # Calculate means
    df_avg = df_train.groupby('implementation')[['fwd_ms', 'bwd_ms']].mean()
    
    # Reorder
    existing_order = [x for x in IMPL_ORDER if x in df_avg.index]
    df_avg = df_avg.reindex(existing_order)

    ax = df_avg.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
    
    plt.title('Training Kernel Breakdown (Level 4 ONLY)', fontsize=14)
    plt.ylabel('Time per Batch (ms)')
    plt.xlabel('Implementation')
    plt.xticks(rotation=0)
    plt.legend(["Forward Pass", "Backward Pass"])
    
    # Annotate bars with total time
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f', label_type='center', color='black', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_kernel_breakdown_lvl4.png"))
    print("   📊 Saved Level 4 Breakdown Chart")




def generate_training_table(df):
    """Generates and saves a detailed Training Performance Table."""
    df_train = df[df['type'] == 'train']
    if df_train.empty: return

    print("\n--- 🏋️  Training Performance Summary ---")
    
    # Aggregation: Group by Implementation AND Level
    summary = df_train.groupby(['wt_levels', 'implementation']).agg({
        'train_throughput': ['mean', 'std'], # Speed
        'vram_mb': ['mean'],                 # Memory cost
        'fwd_ms': ['mean'],                  # Forward speed
        'bwd_ms': ['mean']                   # Backward speed (Critical!)
    }).round(1)
    
    # Print to console
    print(summary)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "training_summary_by_level.csv")
    summary.to_csv(csv_path)
    print(f"\n✅ Saved Training Table to: {csv_path}")    


def generate_inference_table(df):
    """Generates and saves a detailed Inference Performance Table (All Levels)."""
    df_inf = df[df['type'] == 'inference']
    if df_inf.empty: return

    print("\n--- ⚡ Inference Performance Summary (All Levels) ---")
    
    # Aggregation: Group by Level & Implementation
    summary = df_inf.groupby(['wt_levels', 'implementation']).agg({
        'inf_latency': ['mean', 'std'],
        'inf_throughput': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(2)
    
    print(summary)
    
    csv_path = os.path.join(OUTPUT_DIR, "inference_summary_by_level.csv")
    summary.to_csv(csv_path)
    print(f"\n✅ Saved Inference Table to: {csv_path}")    


# (Assuming 'df' is already loaded with your CSV data)

def plot_advanced_analysis(df):
    output_dir = OUTPUT_DIR
    
    # 1. Calculate Total Kernel Time for ALL rows first
    df['total_kernel_ms'] = df['fwd_ms'] + df['bwd_ms']
    
    # --- CRITICAL FIX: Aggregate by Implementation & Level ---
    # We take the mean across seeds to ensure unique rows for the math
    df_avg = df.groupby(['implementation', 'wt_levels'], as_index=False)[
        ['total_kernel_ms', 'fwd_ms', 'bwd_ms']
    ].mean()

    # --- Chart 1. Total Kernel Latency (Using Aggregated Data) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_avg, x='wt_levels', y='total_kernel_ms', hue='implementation',
                 style='implementation', markers=True, dashes=False, linewidth=3,
                 palette='viridis', hue_order=["reference", "cuda", "cuda_opt", "cuda_opt2"])
    
    # Add visual arrow highlighting the "L4 vs L1" win
    # We use safe access .iloc[0] to get the single value
    try:
        ref_l1 = df_avg[(df_avg['implementation']=='reference') & (df_avg['wt_levels']==1)]['total_kernel_ms'].iloc[0]
        opt_l4 = df_avg[(df_avg['implementation']=='cuda_opt2') & (df_avg['wt_levels']==4)]['total_kernel_ms'].iloc[0]
        
        if opt_l4 < ref_l1:
            plt.annotate(
                f"Opt2 (Lvl 4) is Faster\nthan Ref (Lvl 1)!", 
                xy=(4, opt_l4), xytext=(2.5, ref_l1 + 10),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                fontsize=11, fontweight='bold', color='darkred'
            )
            plt.axhline(ref_l1, color='gray', linestyle=':', alpha=0.6)
    except IndexError:
        pass # Skip annotation if data is missing

    plt.title('Total Training Kernel Time (Fwd + Bwd)', fontsize=14)
    plt.ylabel('Time per Batch (ms) [Lower is Better]')
    plt.xlabel('Wavelet Levels')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks([1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_total_kernel_latency.png"))
    print("   📊 Saved Total Kernel Latency Chart")

    # --- Chart 2. Marginal Cost of Depth ---
    # Create the baseline lookup from the AVERAGED dataframe (Unique Index!)
    l1_times = df_avg[df_avg['wt_levels'] == 1].set_index('implementation')['total_kernel_ms']
    
    # Function to subtract baseline safely
    def calculate_added_cost(row):
        base = l1_times.get(row['implementation'])
        if base is not None:
            return row['total_kernel_ms'] - base
        return 0

    df_avg['added_ms'] = df_avg.apply(calculate_added_cost, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_avg[df_avg['wt_levels'] > 1], 
                x='wt_levels', y='added_ms', hue='implementation',
                palette='rocket_r', hue_order=["reference", "cuda_opt2"])
    
    plt.title('Marginal Cost: Extra Time Added by Increasing Depth', fontsize=14)
    plt.ylabel('Additional Milliseconds (vs Level 1)')
    plt.xlabel('Wavelet Levels')
    plt.legend(title="Implementation")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_marginal_cost.png"))
    print("   📊 Saved Marginal Cost Chart")

    # --- Chart 3. Speedup Heatmap ---
    # Use average times for the heatmap
    ref_times = df_avg[df_avg['implementation']=='reference'].set_index('wt_levels')['total_kernel_ms']
    
    def calc_speedup(row):
        ref = ref_times.get(row['wt_levels'])
        if ref and row['total_kernel_ms'] > 0:
            return ref / row['total_kernel_ms']
        return 0
        
    df_avg['speedup'] = df_avg.apply(calc_speedup, axis=1)
    
    # Pivot for Heatmap
    heatmap_data = df_avg.pivot(index='wt_levels', columns='implementation', values='speedup')
    
    # Filter to show only CUDA versions
    cols_to_show = [c for c in ["cuda", "cuda_opt", "cuda_opt2"] if c in heatmap_data.columns]
    if cols_to_show:
        heatmap_data = heatmap_data[cols_to_show]
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Greens", 
                   linewidths=.5, cbar_kws={'label': 'Speedup Factor'})
        plt.title('Training Speedup Factor vs. Reference', fontsize=14)
        plt.ylabel('Wavelet Levels')
        plt.xlabel('Implementation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "7_speedup_heatmap.png"))
        print("   📊 Saved Speedup Heatmap")


def plot_inference_heatmap(df):
    output_dir = OUTPUT_DIR
    
    # 1. Filter for Inference Data
    df_inf = df[df['type'] == 'inference'].copy()
    
    # 2. Aggregate (Mean across seeds if multiple exist)
    df_avg = df_inf.groupby(['implementation', 'wt_levels'], as_index=False)['inf_latency'].mean()

    # 3. Calculate Speedup Factor vs Reference
    # Get reference latency for each level
    ref_map = df_avg[df_avg['implementation'] == 'reference'].set_index('wt_levels')['inf_latency']
    
    def get_speedup(row):
        ref_lat = ref_map.get(row['wt_levels'])
        if ref_lat and row['inf_latency'] > 0:
            return ref_lat / row['inf_latency']
        return 0
        
    df_avg['speedup'] = df_avg.apply(get_speedup, axis=1)

    # 4. Pivot for Heatmap (Rows: Levels, Cols: Impl)
    heatmap_data = df_avg.pivot(index='wt_levels', columns='implementation', values='speedup')
    
    # Select only the CUDA implementations to keep it clean
    cols_to_show = [c for c in ["cuda", "cuda_opt", "cuda_opt2"] if c in heatmap_data.columns]
    
    if cols_to_show:
        heatmap_data = heatmap_data[cols_to_show]
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="Blues", 
                   linewidths=.5, cbar_kws={'label': 'Speedup Factor (x Times Faster)'},
                   annot_kws={"size": 12, "weight": "bold"})
        
        plt.title('Inference Speedup vs. PyTorch Reference', fontsize=14)
        plt.ylabel('Wavelet Levels (Complexity)')
        plt.xlabel('Implementation')
        plt.yticks(rotation=0) 
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, "8_inference_speedup_heatmap.png")
        plt.savefig(save_path)
        print(f"   📊 Saved Inference Heatmap to {save_path}")
    else:
        print("   ⚠️ Skipped Inference Heatmap (No CUDA data found)")        

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- 📥 Processing Data ---")
    df = load_data()
    
    if df.empty:
        print("❌ No data found.")
        return

    # --- 1. Generate Summary Table (Mean ± Std) ---
    print("\n--- 📝 Summary Table (Level 4 Focus) ---")
    # Filter for Level 4 Inference to give a snapshot of "peak performance"
    lvl4_data = df[(df['wt_levels'] == 4) & (df['type'] == 'inference')]
    
    if not lvl4_data.empty:
        summary = lvl4_data.groupby('implementation').agg({
            'inf_latency': ['mean', 'std'],
            'inf_throughput': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(2)
        print(">>> Level 4 Performance Stats:")
        print(summary)
        summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats_lvl4.csv"))
    
    # --- 2. Generate Plots ---
    print("\n--- 🎨 Generating Plots ---")
    plot_scalability_latency(df)
    plot_throughput_comparison(df)
    plot_kernel_breakdown_level4(df)
    generate_training_table(df)
    generate_inference_table(df)
    plot_advanced_analysis(df)
    plot_inference_heatmap(df)
    
    print(f"\n✅ Analysis Complete. Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()