import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
TRAIN_RESULTS_DIR = "./experiment_results"
INFERENCE_RESULTS_DIR = "./inference_reports"
OUTPUT_DIR = "./analysis_output"
IMPL_ORDER = ["reference", "cuda", "cuda_opt", "cuda_opt2"]

def load_data_robust(search_path, type_label):
    data = []
    files = glob.glob(search_path)
    for f in files:
        with open(f, 'r') as file:
            try:
                content = json.load(file)
                if isinstance(content, list): data.extend(content)
                else: data.append(content)
            except: pass
    return pd.DataFrame(data)

def get_combined_df():
    # Load Inference
    inf_df = load_data_robust(os.path.join(INFERENCE_RESULTS_DIR, "*.json"), "INFER")
    if not inf_df.empty:
        # Standardize column name for levels
        # In inference.py we called it 'wt_levels'
        inf_df = inf_df.rename(columns={
            'throughput_imgs_sec': 'throughput',
            'latency_ms': 'latency',
            'accuracy_percent': 'accuracy'
        })

    # Load Training
    train_df = load_data_robust(os.path.join(TRAIN_RESULTS_DIR, "*.json"), "TRAIN")
    if not train_df.empty:
        # In train.py it was 'wy_levels' (typo fix), let's map it to 'wt_levels'
        train_df = train_df.rename(columns={
            'throughput_img_sec': 'throughput', 
            'wy_levels': 'wt_levels'
        })

    # If we have both, combine them (stacking rows, not merging columns)
    # We want a long format: [Implementation, Level, Metric, Type]
    
    # For this specific plot, let's focus on INFERENCE data as it's the most critical
    return inf_df

def plot_scaling(df):
    """Line chart: Levels vs Latency"""
    if df.empty or 'wt_levels' not in df.columns: return

    plt.figure(figsize=(10, 6))
    
    # Calculate Mean per (Implementation, Level)
    df_mean = df.groupby(['implementation', 'wt_levels'])['latency'].mean().reset_index()
    
    sns.lineplot(
        data=df_mean, x='wt_levels', y='latency', hue='implementation', 
        style='implementation', markers=True, dashes=False, linewidth=2.5,
        palette='viridis', hue_order=IMPL_ORDER
    )
    
    plt.title('Scalability: Latency vs. Wavelet Levels', fontsize=14)
    plt.xlabel('Wavelet Levels (More Levels = More Compute)')
    plt.ylabel('Inference Latency (ms)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(df_mean['wt_levels'].unique()) # Ensure we show 1, 2, 3, 4 integers
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_scalability_latency.png"))
    print("   📊 Saved Scalability Chart")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = get_combined_df()
    
    if df.empty:
        print("❌ No data found.")
        return

    print("\n--- 📝 Summary by Level ---")
    # Group by BOTH implementation and level
    summary = df.groupby(['implementation', 'wt_levels'])[['latency', 'accuracy']].mean()
    print(summary.round(2))
    summary.to_csv(os.path.join(OUTPUT_DIR, "scalability_summary.csv"))
    
    print("\n--- 🎨 Generating Plots ---")
    plot_scaling(df)

if __name__ == "__main__":
    main()