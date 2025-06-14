import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

def detect_csv_files():
    """
    Detect all CSV files in the current directory
    """
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found in the current directory.")
        return []
    
    print(f"Found {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    return csv_files

def analyze_csv_structure(filename):
    """
    Analyze the structure of a CSV file to determine the best plotting approach
    """
    try:
        df = pd.read_csv(filename)
        
        info = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'has_time_series': False,
            'data_type': 'unknown'
        }
        
        # Check if it's time series data
        if any(col.lower() in ['time', 'date', 'timestamp', 'epoch'] for col in df.columns):
            info['has_time_series'] = True
            info['data_type'] = 'time_series'
        
        # Determine data type based on structure
        if 'loss' in df.columns:
            info['data_type'] = 'training_loss'
        elif any(col in ['predicted', 'actual'] for col in df.columns):
            info['data_type'] = 'predictions'
        elif len(info['numeric_columns']) > 1:
            info['data_type'] = 'multivariate'
        elif len(info['numeric_columns']) == 1:
            info['data_type'] = 'univariate'
        
        return info, df
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def plot_training_loss(df, filename):
    """
    Plot training loss curves
    """
    plt.figure(figsize=(10, 6))
    
    if 'loss' in df.columns:
        plt.plot(df.index, df['loss'], linewidth=2, label='Training Loss')
        plt.title(f'Training Loss Curve - {filename}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale for better visualization
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_name = f"plot_{filename.replace('.csv', '_loss.png')}"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.show()

def plot_predictions(df, filename):
    """
    Plot prediction vs actual data
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Prediction Analysis - {filename}', fontsize=14, fontweight='bold')
    
    # Scatter plot: Predicted vs Actual
    if 'predicted' in df.columns and 'actual_num' in df.columns:
        colors = ['red' if x == 0 else 'green' for x in df['actual_num']]
        axes[0].scatter(df['actual_num'], df['predicted'], c=colors, alpha=0.7, s=100, edgecolors='black')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
        axes[0].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Decision Threshold')
        axes[0].axvline(x=0.5, color='orange', linestyle=':', alpha=0.7)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predicted vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Age vs Prediction (if age column exists)
    if 'age' in df.columns and 'predicted' in df.columns:
        if 'sex' in df.columns:
            colors = ['blue' if sex == 'male' else 'red' for sex in df['sex']]
            axes[1].scatter(df['age'], df['predicted'], c=colors, alpha=0.7, s=100, edgecolors='black')
            axes[1].set_title('Age vs Prediction (Blue=Male, Red=Female)')
        else:
            axes[1].scatter(df['age'], df['predicted'], alpha=0.7, s=100, edgecolors='black')
            axes[1].set_title('Age vs Prediction')
        
        # Add trend line
        z = np.polyfit(df['age'], df['predicted'], 1)
        p = np.poly1d(z)
        axes[1].plot(df['age'], p(df['age']), "k--", alpha=0.8, linewidth=2)
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Predicted Probability')
        axes[1].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7)
        axes[1].grid(True, alpha=0.3)
    else:
        # Histogram of predictions
        axes[1].hist(df['predicted'], bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Predictions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_name = f"plot_{filename.replace('.csv', '_predictions.png')}"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.show()

def plot_multivariate(df, filename):
    """
    Plot multivariate data with correlation matrix and pairwise relationships
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print(f"Not enough numeric columns in {filename} for multivariate analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Multivariate Analysis - {filename}', fontsize=14, fontweight='bold')
    
    # 1. Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Matrix')
    
    # 2. Line plot of all numeric columns
    for col in numeric_cols:
        axes[0, 1].plot(df.index, df[col], label=col, alpha=0.7, linewidth=2)
    axes[0, 1].set_title('All Numeric Columns Over Index')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot of numeric columns
    df[numeric_cols].boxplot(ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Numeric Columns')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot of first two numeric columns
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        axes[1, 1].scatter(df[col1], df[col2], alpha=0.7, s=50)
        axes[1, 1].set_xlabel(col1)
        axes[1, 1].set_ylabel(col2)
        axes[1, 1].set_title(f'{col1} vs {col2}')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_name = f"plot_{filename.replace('.csv', '_multivariate.png')}"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.show()

def plot_univariate(df, filename):
    """
    Plot univariate data
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print(f"No numeric columns found in {filename}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Univariate Analysis - {filename}', fontsize=14, fontweight='bold')
    
    col = numeric_cols[0]
    
    # 1. Line plot
    axes[0, 0].plot(df.index, df[col], linewidth=2, alpha=0.8)
    axes[0, 0].set_title(f'{col} Over Index')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel(col)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[0, 1].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'Distribution of {col}')
    axes[0, 1].set_xlabel(col)
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot
    axes[1, 0].boxplot(df[col].dropna())
    axes[1, 0].set_title(f'Box Plot of {col}')
    axes[1, 0].set_ylabel(col)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_data = np.sort(df[col].dropna())
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 1].plot(sorted_data, cumulative, linewidth=2)
    axes[1, 1].set_title(f'Cumulative Distribution of {col}')
    axes[1, 1].set_xlabel(col)
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_name = f"plot_{filename.replace('.csv', '_univariate.png')}"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.show()

def plot_generic_data(df, filename):
    """
    Generic plotting for any CSV data
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    n_plots = min(4, len(numeric_cols) + len(categorical_cols))
    if n_plots == 0:
        print(f"No plottable data found in {filename}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Generic Data Analysis - {filename}', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot numeric columns
    for i, col in enumerate(numeric_cols[:2]):
        if plot_idx >= 4:
            break
        axes[plot_idx].plot(df.index, df[col], linewidth=2, alpha=0.8)
        axes[plot_idx].set_title(f'{col} Over Index')
        axes[plot_idx].set_xlabel('Index')
        axes[plot_idx].set_ylabel(col)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot categorical columns
    for i, col in enumerate(categorical_cols[:2]):
        if plot_idx >= 4:
            break
        value_counts = df[col].value_counts()
        axes[plot_idx].bar(value_counts.index, value_counts.values, alpha=0.7)
        axes[plot_idx].set_title(f'Distribution of {col}')
        axes[plot_idx].set_xlabel(col)
        axes[plot_idx].set_ylabel('Count')
        axes[plot_idx].tick_params(axis='x', rotation=45)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_name = f"plot_{filename.replace('.csv', '_generic.png')}"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.show()

def create_summary_report(csv_files):
    """
    Create a summary report of all CSV files
    """
    print("\n" + "="*80)
    print("CSV DATA ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    for filename in csv_files:
        info, df = analyze_csv_structure(filename)
        if info is None:
            continue
        
        print(f"\n📄 FILE: {filename}")
        print(f"   ├── Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        print(f"   ├── Data type: {info['data_type']}")
        print(f"   ├── Numeric columns: {len(info['numeric_columns'])}")
        print(f"   ├── Categorical columns: {len(info['categorical_columns'])}")
        print(f"   └── Columns: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")
        
        # Show basic statistics for numeric columns
        if info['numeric_columns']:
            print(f"   📊 Basic Statistics:")
            for col in info['numeric_columns'][:3]:  # Show first 3 numeric columns
                stats = df[col].describe()
                print(f"      {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    print("\n" + "="*80)

def main():
    """
    Main function to detect and plot all CSV files
    """
    print("🔍 CSV Data Plotting Utility")
    print("="*50)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Detect CSV files
    csv_files = detect_csv_files()
    if not csv_files:
        return
    
    # Create summary report
    create_summary_report(csv_files)
    
    print(f"\n📊 Creating plots for {len(csv_files)} CSV files...")
    print("-" * 50)
    
    # Process each CSV file
    for filename in csv_files:
        print(f"\nProcessing: {filename}")
        
        info, df = analyze_csv_structure(filename)
        if info is None:
            continue
        
        # Choose appropriate plotting function based on data type
        try:
            if info['data_type'] == 'training_loss':
                plot_training_loss(df, filename)
            elif info['data_type'] == 'predictions':
                plot_predictions(df, filename)
            elif info['data_type'] == 'multivariate':
                plot_multivariate(df, filename)
            elif info['data_type'] == 'univariate':
                plot_univariate(df, filename)
            else:
                plot_generic_data(df, filename)
                
        except Exception as e:
            print(f"Error plotting {filename}: {e}")
            # Fallback to generic plotting
            try:
                plot_generic_data(df, filename)
            except Exception as e2:
                print(f"Failed to create any plot for {filename}: {e2}")
    
    print(f"\n✅ Plotting complete! Check the generated PNG files.")

if __name__ == "__main__":
    main() 