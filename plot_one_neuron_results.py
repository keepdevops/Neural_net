import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_one_neuron_results():
    """
    Plot comprehensive results from the one neuron neural network training
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('One Neuron Neural Network - Training Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss curves for both models
    try:
        loss_fries = pd.read_csv('loss_curve_will_buy_fries.csv')
        loss_hamburger = pd.read_csv('loss_curve_will_buy_hamburger.csv')
        
        axes[0, 0].plot(loss_fries['loss'], label='Fries Model', linewidth=2, alpha=0.8)
        axes[0, 0].plot(loss_hamburger['loss'], label='Hamburger Model', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Training Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Log scale to better see convergence
        
    except FileNotFoundError:
        axes[0, 0].text(0.5, 0.5, 'Loss curve files not found', ha='center', va='center')
        axes[0, 0].set_title('Training Loss Curves - Data Missing')
    
    # 2. Predictions vs Actual for Fries
    try:
        pred_fries = pd.read_csv('predictions_will_buy_fries.csv')
        
        # Create scatter plot
        colors = ['red' if actual == 0 else 'green' for actual in pred_fries['actual_num']]
        scatter = axes[0, 1].scatter(pred_fries['actual_num'], pred_fries['predicted'], 
                                   c=colors, alpha=0.7, s=100, edgecolors='black')
        
        # Add perfect prediction line
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
        axes[0, 1].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Decision Threshold')
        axes[0, 1].axvline(x=0.5, color='orange', linestyle=':', alpha=0.7)
        
        axes[0, 1].set_title('Fries Model: Predictions vs Actual', fontweight='bold')
        axes[0, 1].set_xlabel('Actual (0=No Fries, 1=Fries)')
        axes[0, 1].set_ylabel('Predicted Probability')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
    except FileNotFoundError:
        axes[0, 1].text(0.5, 0.5, 'Fries predictions not found', ha='center', va='center')
        axes[0, 1].set_title('Fries Predictions - Data Missing')
    
    # 3. Predictions vs Actual for Hamburger
    try:
        pred_hamburger = pd.read_csv('predictions_will_buy_hamburger.csv')
        
        # Create scatter plot
        colors = ['red' if actual == 0 else 'green' for actual in pred_hamburger['actual_num']]
        scatter = axes[0, 2].scatter(pred_hamburger['actual_num'], pred_hamburger['predicted'], 
                                   c=colors, alpha=0.7, s=100, edgecolors='black')
        
        # Add perfect prediction line
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
        axes[0, 2].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Decision Threshold')
        axes[0, 2].axvline(x=0.5, color='orange', linestyle=':', alpha=0.7)
        
        axes[0, 2].set_title('Hamburger Model: Predictions vs Actual', fontweight='bold')
        axes[0, 2].set_xlabel('Actual (0=No Hamburger, 1=Hamburger)')
        axes[0, 2].set_ylabel('Predicted Probability')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
    except FileNotFoundError:
        axes[0, 2].text(0.5, 0.5, 'Hamburger predictions not found', ha='center', va='center')
        axes[0, 2].set_title('Hamburger Predictions - Data Missing')
    
    # 4. Model Weights Comparison
    try:
        weights_fries = np.loadtxt('weights_fries.txt')[-1]  # Get last weights
        weights_hamburger = np.loadtxt('weights_hamburger.txt')[-1]  # Get last weights
        
        feature_names = ['Sex Weight', 'Age Weight']
        x_pos = np.arange(len(feature_names))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x_pos - width/2, weights_fries, width, 
                              label='Fries Model', alpha=0.8, color='skyblue')
        bars2 = axes[1, 0].bar(x_pos + width/2, weights_hamburger, width, 
                              label='Hamburger Model', alpha=0.8, color='lightcoral')
        
        axes[1, 0].set_title('Model Weights Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Weight Value')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(feature_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars1, weights_fries):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        for bar, value in zip(bars2, weights_hamburger):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
    except FileNotFoundError:
        axes[1, 0].text(0.5, 0.5, 'Weight files not found', ha='center', va='center')
        axes[1, 0].set_title('Model Weights - Data Missing')
    
    # 5. Age vs Prediction Probability (Fries)
    try:
        axes[1, 1].scatter(pred_fries['age'], pred_fries['predicted'], 
                          c=['blue' if sex == 'male' else 'red' for sex in pred_fries['sex']], 
                          alpha=0.7, s=100, edgecolors='black')
        
        # Add trend line
        z = np.polyfit(pred_fries['age'], pred_fries['predicted'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(pred_fries['age'], p(pred_fries['age']), "k--", alpha=0.8, linewidth=2)
        
        axes[1, 1].set_title('Fries: Age vs Prediction (Blue=Male, Red=Female)', fontweight='bold')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Predicted Probability')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.5, color='orange', linestyle=':', alpha=0.7)
        
    except (FileNotFoundError, NameError):
        axes[1, 1].text(0.5, 0.5, 'Age analysis data not available', ha='center', va='center')
        axes[1, 1].set_title('Age Analysis - Data Missing')
    
    # 6. Model Performance Summary
    try:
        # Calculate accuracy for both models
        fries_accuracy = np.mean((pred_fries['predicted'] >= 0.5) == pred_fries['actual_num'])
        hamburger_accuracy = np.mean((pred_hamburger['predicted'] >= 0.5) == pred_hamburger['actual_num'])
        
        # Calculate final losses
        final_loss_fries = loss_fries['loss'].iloc[-1]
        final_loss_hamburger = loss_hamburger['loss'].iloc[-1]
        
        # Create summary table
        metrics = ['Accuracy', 'Final Loss']
        fries_values = [fries_accuracy, final_loss_fries]
        hamburger_values = [hamburger_accuracy, final_loss_hamburger]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[1, 2].bar(x_pos - width/2, fries_values, width, 
                              label='Fries Model', alpha=0.8, color='skyblue')
        bars2 = axes[1, 2].bar(x_pos + width/2, hamburger_values, width, 
                              label='Hamburger Model', alpha=0.8, color='lightcoral')
        
        axes[1, 2].set_title('Model Performance Summary', fontweight='bold')
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, fries_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        for bar, value in zip(bars2, hamburger_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
    except (FileNotFoundError, NameError):
        axes[1, 2].text(0.5, 0.5, 'Performance data not available', ha='center', va='center')
        axes[1, 2].set_title('Performance Summary - Data Missing')
    
    plt.tight_layout()
    plt.savefig('one_neuron_network_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive analysis plot: one_neuron_network_analysis.png")
    plt.show()

def print_summary_report():
    """
    Print a detailed summary report of the training results
    """
    print("\n" + "="*80)
    print("ONE NEURON NEURAL NETWORK - TRAINING SUMMARY REPORT")
    print("="*80)
    
    try:
        # Load data
        pred_fries = pd.read_csv('predictions_will_buy_fries.csv')
        pred_hamburger = pd.read_csv('predictions_will_buy_hamburger.csv')
        loss_fries = pd.read_csv('loss_curve_will_buy_fries.csv')
        loss_hamburger = pd.read_csv('loss_curve_will_buy_hamburger.csv')
        weights_fries = np.loadtxt('weights_fries.txt')[-1]
        weights_hamburger = np.loadtxt('weights_hamburger.txt')[-1]
        
        print(f"\n📊 DATASET INFORMATION:")
        print(f"   Total samples: {len(pred_fries)}")
        print(f"   Features: Sex (0=Male, 1=Female), Age")
        print(f"   Targets: Fries purchase, Hamburger purchase")
        
        print(f"\n🧠 MODEL ARCHITECTURE:")
        print(f"   Input neurons: 2 (sex, age)")
        print(f"   Hidden neurons: 0 (single neuron network)")
        print(f"   Output neurons: 1 (binary classification)")
        print(f"   Activation function: Sigmoid")
        print(f"   Loss function: Mean Squared Error (MSE)")
        
        print(f"\n📈 TRAINING RESULTS:")
        print(f"   Training epochs: {len(loss_fries)}")
        
        print(f"\n   FRIES MODEL:")
        print(f"   ├── Final loss: {loss_fries['loss'].iloc[-1]:.6f}")
        print(f"   ├── Sex weight: {weights_fries[0]:.4f}")
        print(f"   ├── Age weight: {weights_fries[1]:.4f}")
        fries_accuracy = np.mean((pred_fries['predicted'] >= 0.5) == pred_fries['actual_num'])
        print(f"   └── Accuracy: {fries_accuracy:.4f} ({fries_accuracy*100:.1f}%)")
        
        print(f"\n   HAMBURGER MODEL:")
        print(f"   ├── Final loss: {loss_hamburger['loss'].iloc[-1]:.6f}")
        print(f"   ├── Sex weight: {weights_hamburger[0]:.4f}")
        print(f"   ├── Age weight: {weights_hamburger[1]:.4f}")
        hamburger_accuracy = np.mean((pred_hamburger['predicted'] >= 0.5) == pred_hamburger['actual_num'])
        print(f"   └── Accuracy: {hamburger_accuracy:.4f} ({hamburger_accuracy*100:.1f}%)")
        
        print(f"\n🔍 MODEL INTERPRETATION:")
        print(f"   FRIES MODEL INSIGHTS:")
        if weights_fries[0] > 0:
            print(f"   ├── Females are MORE likely to buy fries (sex weight: +{weights_fries[0]:.4f})")
        else:
            print(f"   ├── Males are MORE likely to buy fries (sex weight: {weights_fries[0]:.4f})")
        
        if weights_fries[1] > 0:
            print(f"   └── Older people are MORE likely to buy fries (age weight: +{weights_fries[1]:.4f})")
        else:
            print(f"   └── Younger people are MORE likely to buy fries (age weight: {weights_fries[1]:.4f})")
        
        print(f"\n   HAMBURGER MODEL INSIGHTS:")
        if weights_hamburger[0] > 0:
            print(f"   ├── Females are MORE likely to buy hamburgers (sex weight: +{weights_hamburger[0]:.4f})")
        else:
            print(f"   ├── Males are MORE likely to buy hamburgers (sex weight: {weights_hamburger[0]:.4f})")
        
        if weights_hamburger[1] > 0:
            print(f"   └── Older people are MORE likely to buy hamburgers (age weight: +{weights_hamburger[1]:.4f})")
        else:
            print(f"   └── Younger people are MORE likely to buy hamburgers (age weight: {weights_hamburger[1]:.4f})")
        
        print(f"\n📋 SAMPLE PREDICTIONS:")
        print(f"   FRIES MODEL - Sample predictions:")
        for i in range(min(5, len(pred_fries))):
            row = pred_fries.iloc[i]
            decision = 'WILL BUY' if row['predicted'] >= 0.5 else "WON'T BUY"
            print(f"   ├── {row['sex'].title()}, Age {int(row['age'])}: {row['predicted']:.4f} prob → {decision} (Actual: {row['actual']})")
        
        print(f"\n   HAMBURGER MODEL - Sample predictions:")
        for i in range(min(5, len(pred_hamburger))):
            row = pred_hamburger.iloc[i]
            decision = 'WILL BUY' if row['predicted'] >= 0.5 else "WON'T BUY"
            print(f"   ├── {row['sex'].title()}, Age {int(row['age'])}: {row['predicted']:.4f} prob → {decision} (Actual: {row['actual']})")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files: {e}")
        print("   Please run the neural network training first.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("One Neuron Neural Network Results Visualization")
    print("=" * 50)
    
    # Create plots
    plot_one_neuron_results()
    
    # Print summary report
    print_summary_report() 