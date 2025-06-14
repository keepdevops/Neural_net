import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_data():
    """Load all the saved data from CSV files"""
    try:
        # Load training history
        loss_history = np.loadtxt("loss_history.csv", delimiter=",")
        accuracy_history = np.loadtxt("accuracy_history.csv", delimiter=",")
        
        # Load predictions and labels
        final_predictions = np.loadtxt("final_predictions.csv", delimiter=",")
        actual_labels = np.loadtxt("actual_labels.csv", delimiter=",")
        
        # Load model weights
        W1 = np.loadtxt("W1.csv", delimiter=",")
        W2 = np.loadtxt("W2.csv", delimiter=",")
        
        # Load input data
        input_data = np.loadtxt("input_data.csv", delimiter=",")
        
        return {
            'loss_history': loss_history,
            'accuracy_history': accuracy_history,
            'final_predictions': final_predictions.flatten(),
            'actual_labels': actual_labels.flatten(),
            'W1': W1,
            'W2': W2,
            'input_data': input_data
        }
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV file: {e}")
        print("Please run the neural network training first to generate the data files.")
        sys.exit(1)

def create_comprehensive_plots(data):
    """Create comprehensive visualization of neural network training and results"""
    
    # Create main figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Neural Network Training Analysis', fontsize=20, fontweight='bold')
    
    # 1. Loss curve
    axes[0, 0].plot(data['loss_history'], 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#f8f9fa')
    
    # Add loss statistics
    final_loss = data['loss_history'][-1]
    min_loss = np.min(data['loss_history'])
    axes[0, 0].axhline(y=min_loss, color='r', linestyle='--', alpha=0.7, label=f'Min Loss: {min_loss:.4f}')
    axes[0, 0].text(0.02, 0.98, f'Final Loss: {final_loss:.4f}', transform=axes[0, 0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 0].legend()
    
    # 2. Accuracy curve
    axes[0, 1].plot(data['accuracy_history'], 'g-', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_facecolor('#f8f9fa')
    
    # Add accuracy statistics
    final_accuracy = data['accuracy_history'][-1]
    max_accuracy = np.max(data['accuracy_history'])
    axes[0, 1].axhline(y=max_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Max Accuracy: {max_accuracy:.4f}')
    axes[0, 1].text(0.02, 0.02, f'Final Accuracy: {final_accuracy:.4f}', transform=axes[0, 1].transAxes, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 1].legend()
    
    # 3. Predictions vs Actual
    axes[0, 2].scatter(data['actual_labels'], data['final_predictions'], alpha=0.8, s=150, 
                      c=['red' if x == 0 else 'green' for x in data['actual_labels']], edgecolors='black')
    axes[0, 2].plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    axes[0, 2].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Decision Threshold')
    axes[0, 2].axvline(x=0.5, color='orange', linestyle=':', alpha=0.7)
    axes[0, 2].set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Actual Values')
    axes[0, 2].set_ylabel('Predicted Values')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_facecolor('#f8f9fa')
    axes[0, 2].legend()
    
    # 4. Feature weights visualization (W1)
    im1 = axes[1, 0].imshow(data['W1'].T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    axes[1, 0].set_title('Input-to-Hidden Weights (W1)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Input Features')
    axes[1, 0].set_ylabel('Hidden Neurons')
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
    axes[1, 0].set_yticks(range(data['W1'].shape[1]))
    axes[1, 0].set_yticklabels([f'Hidden {i+1}' for i in range(data['W1'].shape[1])])
    cbar1 = plt.colorbar(im1, ax=axes[1, 0])
    cbar1.set_label('Weight Value')
    
    # 5. Hidden-to-output weights (W2)
    colors = ['red' if w < 0 else 'green' for w in data['W2'].flatten()]
    bars = axes[1, 1].bar(range(len(data['W2'].flatten())), data['W2'].flatten(), 
                         color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Hidden-to-Output Weights (W2)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Hidden Neuron')
    axes[1, 1].set_ylabel('Weight Value')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_facecolor('#f8f9fa')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, data['W2'].flatten()):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # 6. Sample predictions with confidence
    sample_indices = range(len(data['actual_labels']))
    width = 0.35
    x_pos = np.arange(len(sample_indices))
    
    bars1 = axes[1, 2].bar(x_pos - width/2, data['final_predictions'], width, 
                          alpha=0.8, label='Predicted', color='blue', edgecolor='black')
    bars2 = axes[1, 2].bar(x_pos + width/2, data['actual_labels'], width, 
                          alpha=0.8, label='Actual', color='red', edgecolor='black')
    
    axes[1, 2].set_title('Sample Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Sample Index')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f'S{i+1}' for i in sample_indices])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('neural_network_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: neural_network_comprehensive_analysis.png")
    plt.show()

def create_detailed_analysis_plots(data):
    """Create additional detailed analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Neural Network Analysis', fontsize=18, fontweight='bold')
    
    # 1. Loss and Accuracy on same plot with dual y-axis
    ax1 = axes[0, 0]
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    line1 = ax1.plot(data['loss_history'], color=color1, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color2)
    line2 = ax2.plot(data['accuracy_history'], color=color2, linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.set_title('Training Progress (Loss & Accuracy)', fontweight='bold')
    
    # 2. Residuals plot
    residuals = data['actual_labels'] - data['final_predictions']
    axes[0, 1].scatter(range(len(residuals)), residuals, alpha=0.8, s=150, 
                      c=['red' if r > 0 else 'blue' for r in residuals], edgecolors='black')
    axes[0, 1].axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Residuals (Actual - Predicted)', fontweight='bold')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_facecolor('#f8f9fa')
    
    # Add residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    axes[0, 1].text(0.02, 0.98, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Histogram of predictions
    axes[1, 0].hist(data['final_predictions'], bins=15, alpha=0.7, color='skyblue', 
                   edgecolor='black', density=True)
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[1, 0].axvline(x=np.mean(data['final_predictions']), color='green', linestyle=':', 
                      linewidth=2, label=f'Mean: {np.mean(data["final_predictions"]):.3f}')
    axes[1, 0].set_title('Distribution of Predictions', fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor('#f8f9fa')
    
    # 4. Feature importance (based on average absolute weights)
    feature_importance = np.mean(np.abs(data['W1']), axis=1)
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    bars = axes[1, 1].bar(feature_names, feature_importance, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('Feature Importance (Avg |Weight|)', fontweight='bold')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Average |Weight|')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for bar, value in zip(bars, feature_importance):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('neural_network_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: neural_network_detailed_analysis.png")
    plt.show()

def create_classification_report(data):
    """Create a detailed classification report"""
    print("\n" + "="*60)
    print("NEURAL NETWORK CLASSIFICATION REPORT")
    print("="*60)
    
    print(f"Final Training Loss: {data['loss_history'][-1]:.6f}")
    print(f"Final Training Accuracy: {data['accuracy_history'][-1]:.4f}")
    print(f"Total Training Epochs: {len(data['loss_history'])}")
    
    print(f"\nModel Architecture:")
    print(f"  Input Features: {data['W1'].shape[0]}")
    print(f"  Hidden Neurons: {data['W1'].shape[1]}")
    print(f"  Output Neurons: {data['W2'].shape[1]}")
    
    print(f"\nSample-by-Sample Results:")
    print("-" * 60)
    print(f"{'Sample':<8} {'Predicted':<12} {'Actual':<8} {'Class':<12} {'Confidence':<12}")
    print("-" * 60)
    
    for i, (pred, actual) in enumerate(zip(data['final_predictions'], data['actual_labels'])):
        predicted_class = "PASS" if pred >= 0.5 else "FAIL"
        actual_class = "PASS" if actual == 1 else "FAIL"
        confidence = pred if pred >= 0.5 else 1 - pred
        status = "✓" if predicted_class == actual_class else "✗"
        
        print(f"{i+1:<8} {pred:<12.4f} {actual:<8.0f} {predicted_class:<12} {confidence:<12.4f} {status}")
    
    # Calculate metrics
    predictions_binary = (data['final_predictions'] >= 0.5).astype(int)
    correct_predictions = np.sum(predictions_binary == data['actual_labels'])
    total_predictions = len(data['actual_labels'])
    
    print("-" * 60)
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Final Accuracy: {correct_predictions/total_predictions:.4f}")

def main():
    """Main function to run all plotting functions"""
    print("Neural Network Plotting Program")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("loss_history.csv"):
        print("Warning: Training data files not found in current directory.")
        print("Make sure you've run the neural network training first.")
        return
    
    # Load data
    print("Loading training data...")
    data = load_data()
    print("Data loaded successfully!")
    
    # Create plots
    print("\nCreating comprehensive analysis plots...")
    create_comprehensive_plots(data)
    
    print("\nCreating detailed analysis plots...")
    create_detailed_analysis_plots(data)
    
    # Print classification report
    create_classification_report(data)
    
    print(f"\nPlotting complete! Generated files:")
    print("  - neural_network_comprehensive_analysis.png")
    print("  - neural_network_detailed_analysis.png")

if __name__ == "__main__":
    main() 