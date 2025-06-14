import numpy as np
import os
from duckdb_to_csv_converter import (
    extract_tsla_training_data,
    load_features_and_labels_csv,
    save_arrays_to_csv
)

# Configuration for data source
USE_DUCKDB = True
USE_GUI = False  # Disable GUI for automated processing
CREATE_TRAINING_CSV = True  # Create training CSV from DuckDB
FEATURES_CSV = "tsla_features.csv"
LABELS_CSV = "tsla_labels.csv"
BINARY_LABELS = True  # Convert close prices to binary up/down labels
YEARS_BACK = 2

# DuckDB configuration
DUCKDB_PATH = "/Users/porupine/redline/data/tsla.us_data.duckdb"  # Your TSLA data

#1. Define the dataset: 
#-----------------------#

X = None
y = None

if CREATE_TRAINING_CSV and DUCKDB_PATH:
    print("Creating training CSV from DuckDB...")
    X, y = extract_tsla_training_data(
        db_path=DUCKDB_PATH,
        features_csv=FEATURES_CSV,
        labels_csv=LABELS_CSV,
        binary_labels=BINARY_LABELS,
        years_back=YEARS_BACK
    )

# Fall back to loading existing CSV if DuckDB processing failed
if X is None and os.path.exists(FEATURES_CSV) and os.path.exists(LABELS_CSV):
    print(f"Loading existing training CSV: {FEATURES_CSV}, {LABELS_CSV}")
    X, y = load_features_and_labels_csv(FEATURES_CSV, LABELS_CSV)

# Final fallback to hardcoded data
if X is None:
    print("Using hardcoded dataset...")
    X = np.array([
        [1,5,60,2],
        [3,6,75,3],
        [5,7,95,4],
        [0,4,50,1],
        [5,8,90,4],
        [2,6,75,2],
        [6,7,87,3],
        [5,8,45,4],
        [1,1,0,4],
        [5,9,91,4]
    ],dtype=np.float32)

    y = np.array([[0],[0],[1],[0],[1],[0],[1],[0],[1],[1]],dtype=np.float32)
    
    # Save hardcoded data to CSV for consistency
    save_arrays_to_csv(X, y, "hardcoded_features.csv", "hardcoded_labels.csv")

# Display final dataset info
print(f"\n=== Final Training Dataset ===")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Features: {X.shape[1]} columns")
print(f"Samples: {X.shape[0]} rows")

# Show sample data
print(f"\nSample features (X):")
print(X[:5])
print(f"\nSample labels (y):")
print(y[:5].flatten())

# Show label distribution
if y is not None:
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution: {dict(zip(unique_labels, counts))}")

#-----------------------#
# 2. Normalize the input features(zero mean, unit variance)
#-----------------------#

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
# Avoid division by zero
X_std = np.where(X_std == 0, 1, X_std)
X_norm = (X - X_mean) / X_std

print(f"\nNormalization applied:")
print(f"  Mean: {X_mean}")
print(f"  Std: {X_std}")

#----------------------#
#3. Define activation functions
#----------------------#

def sigmoid(x):
    return 1/(1 + np.exp(-np.clip(x, -250, 250))) #Prevent overflow

def sigmoid_derivative(x):
    return x * (1 - x)

#---------------------#
#4. Initialize network architecture and weights
#--------------------#
input_size = X.shape[1] #Dynamic based on actual features
hidden_size = max(8, input_size * 2) #Increased for more complex patterns
output_size = 1

print(f"\nNeural Network Architecture:")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Output size: {output_size}")

np.random.seed(42)
# Xavier initialization
W1 = np.random.randn(input_size,hidden_size) / np.sqrt(input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size,output_size) / np.sqrt(hidden_size)
b2 = np.zeros((1,output_size))

#--------------------#
#5. Set training parameters
#-------------------#
epochs = 10000
learning_rate = 0.01

print(f"\nTraining Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Starting training...")

#-------------------#
#6. Train network using backpropagation
#-----------------#
print(f"\nStarting training...")

prev_loss = float('inf')
patience = 200
patience_counter = 0
best_loss = float('inf')

for epoch in range(epochs): 
    # Forward pass
    z1 = np.dot(X_norm, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Compute loss and gradients
    error = y - a2
    loss = np.mean(error**2)
   
    d_output = error * sigmoid_derivative(a2)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(a1)
   
    # Update weights and biases
    W2 += a1.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X_norm.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Track best loss
    if loss < best_loss:
        best_loss = loss
    
    # Progress monitoring
    if epoch % 1000 == 0:
        accuracy = np.mean((a2 > 0.5) == (y > 0.5)) * 100
        print(f"Epoch {epoch:5d}, Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%")
    
    # Early stopping
    if loss < prev_loss - 1e-7:
        prev_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
   
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Final evaluation
final_predictions = sigmoid(np.dot(sigmoid(np.dot(X_norm, W1) + b1), W2) + b2)
final_accuracy = np.mean((final_predictions > 0.5) == (y > 0.5)) * 100

print(f"\n=== Training Complete ===")
print(f"Final loss: {loss:.6f}")
print(f"Best loss: {best_loss:.6f}")
print(f"Final accuracy: {final_accuracy:.2f}%")
print(f"Total epochs: {epoch + 1}")

#-------#
#7. Save model parameters
#-----#
np.savetxt("W1.csv", W1, delimiter=",", fmt="%.6f")
np.savetxt("b1.csv", b1, delimiter=",", fmt="%.6f")
np.savetxt("W2.csv", W2, delimiter=",", fmt="%.6f")
np.savetxt("b2.csv", b2, delimiter=",", fmt="%.6f")
np.savetxt("X_mean.csv", X_mean.reshape(1,-1), delimiter=",", fmt="%.6f")
np.savetxt("X_std.csv", X_std.reshape(1,-1), delimiter=",", fmt="%.6f")

print(f"\nModel parameters saved to CSV files")
print(f"Training data CSV files: {FEATURES_CSV}, {LABELS_CSV}")

# Optional: Launch GUI for data visualization and analysis
USE_GUI_AFTER_TRAINING = False  # Set to True to launch GUI after training

if USE_GUI_AFTER_TRAINING:
    print("\nLaunching data visualization GUI...")
    try:
        from data_visualization_gui import launch_data_visualization_gui
        launch_data_visualization_gui()
    except ImportError:
        print("GUI module not available. Install required packages: matplotlib, seaborn")
else:
    print("\nTo launch the data visualization GUI, run:")
    print("python data_visualization_gui.py") 