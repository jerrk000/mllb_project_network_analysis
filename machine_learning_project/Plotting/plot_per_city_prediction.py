"""
Visualize per-city prediction performance from held-out evaluation.

This script creates insightful visualizations showing:
1. Which cities are easy/hard to predict for each model
2. Comparison of model performance across cities
3. City-level prediction patterns and outliers

Usage:
python plot_city_performance.py --results-dir modeling_outputs_with_tuning
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
from typing import Dict, List, Tuple

# Set better plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_prepare_data(results_dir: str = "/modeling_outputs_with_tuning") -> pd.DataFrame:
    """
    Load city-heldout scores and prepare for visualization.
    """
    results_path = Path(results_dir)
    city_scores_path = results_path / "city_heldout_scores.csv"
    
    if not city_scores_path.exists():
        raise FileNotFoundError(f"Could not find {city_scores_path}")
    
    # Load data
    df = pd.read_csv(city_scores_path)
    
    # Basic validation
    required_cols = ['place', 'r2', 'mse', 'model', 'n_test']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Clean data
    df = df.dropna(subset=['r2', 'mse'])
    df['place'] = df['place'].astype(str)
    
    # Add RMSE for easier interpretation
    df['rmse'] = np.sqrt(df['mse'])
    
    # Calculate z-scores for outlier detection
    for model in df['model'].unique():
        mask = df['model'] == model
        df.loc[mask, 'r2_zscore'] = (df.loc[mask, 'r2'] - df.loc[mask, 'r2'].mean()) / df.loc[mask, 'r2'].std()
        df.loc[mask, 'mse_zscore'] = (df.loc[mask, 'mse'] - df.loc[mask, 'mse'].mean()) / df.loc[mask, 'mse'].std()
    
    return df


def plot_city_performance_barplot(df: pd.DataFrame, output_path: Path = "/modeling_outputs_with_tuning"):
    """
    Create bar plot showing R² for each city, grouped by model.
    """
    # Order cities by average R² across models
    city_avg_r2 = df.groupby('place')['r2'].mean().sort_values()
    city_order = city_avg_r2.index.tolist()
    
    # Order models by average R²
    model_avg_r2 = df.groupby('model')['r2'].mean().sort_values(ascending=False)
    model_order = model_avg_r2.index.tolist()
    
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Set up positions for grouped bars
    n_cities = len(city_order)
    n_models = len(model_order)
    bar_width = 0.8 / n_models
    
    # Color palette for models
    model_colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for i, model in enumerate(model_order):
        model_data = df[df['model'] == model]
        # Reorder to match city_order
        model_data = model_data.set_index('place').reindex(city_order).reset_index()
        
        # Calculate positions
        positions = np.arange(n_cities) + i * bar_width - (n_models - 1) * bar_width / 2
        
        bars = ax1.bar(positions, model_data['r2'], 
                      width=bar_width * 0.9,
                      label=model,
                      color=model_colors[i],
                      edgecolor='black',
                      linewidth=0.5,
                      alpha=0.8)
        
        # Add value labels on top of bars
        for bar, r2 in zip(bars, model_data['r2']):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{r2:.2f}', ha='center', va='bottom', 
                        fontsize=8, rotation=90)
    
    ax1.set_xlabel('City', fontsize=12)
    ax1.set_ylabel('R² Score (Leave-One-City-Out)', fontsize=12)
    ax1.set_title('City-Level Prediction Performance by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(np.arange(n_cities))
    ax1.set_xticklabels(city_order, rotation=45, ha='right', fontsize=10)
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add horizontal lines at key R² levels
    for r2_level in [0.3, 0.5, 0.7]:
        ax1.axhline(y=r2_level, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'city_performance_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved bar plot to: {output_path / 'city_performance_barplot.png'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize city-level prediction performance')
    parser.add_argument('--results-dir', type=str, default='modeling_outputs_with_tuning',
                       help='Directory containing modeling results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: results_dir/city_plots)')
    
    args = parser.parse_args()
    
    # Set up directories
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return
    
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = results_path / 'city_performance_plots'
    
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading data from: {results_path}")
    print(f"Saving plots to: {output_path}")
    
    try:
        # Load and process data
        df = load_and_prepare_data(args.results_dir)
        
        print(f"\nData loaded successfully:")
        print(f"  Cities: {len(df['place'].unique())}")
        print(f"  Models: {', '.join(df['model'].unique())}")
        print(f"  Total observations: {len(df)}")
        
        # Create visualizations
        plot_city_performance_barplot(df, output_path)
        
        print("\n" + "="*70)
        print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nCheck these files in {output_path}:")
        print("  1. city_performance_barplot.png - Detailed bar plot")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()