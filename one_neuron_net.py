# Single-Neuron Neural Network for Binary Classification
# This program implements a complete neural network training and prediction system
# with detailed gradient analysis for educational purposes

import numpy as np
import pandas as pd
import sys

# ================================================================================================
# GLOBAL ACTIVATION FUNCTIONS
# ================================================================================================

def sigmoid(x):
    """
    SIGMOID FUNCTION DEFINITION:
    
    Mathematical Formula: σ(z) = 1 / (1 + e^(-z))
    
    PURPOSE: 
    - Maps any real number (scalar or vector) to range (0, 1)
    - Smooth, differentiable function
    - Output can be interpreted as probability for binary classification
    
    INPUT: z (can be scalar or vector)
    OUTPUT: σ(z) (same shape as input, values between 0 and 1)
    
    PROPERTIES:
    - σ(0) = 0.5 (neutral point)
    - σ(+∞) → 1 (approaches 1 for large positive values)
    - σ(-∞) → 0 (approaches 0 for large negative values)
    - Smooth S-shaped curve
    """
    # Echo: Computing sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    SIGMOID DERIVATIVE DEFINITION:
    
    Mathematical Formula: σ'(z) = σ(z) * (1 - σ(z))
    
    PURPOSE: 
    - Used in backward propagation to compute gradients
    - Tells us how much the sigmoid output changes for small changes in input
    - Essential for the chain rule in backpropagation
    
    INPUT: z (same as sigmoid input)
    OUTPUT: derivative value (how steep the sigmoid curve is at point z)
    
    PROPERTIES:
    - Maximum value: 0.25 (at z = 0)
    - Approaches 0 for very large positive or negative z
    - Always positive (sigmoid is always increasing)
    """
    # Echo: Computing sigmoid derivative for backpropagation
    s = sigmoid(x)
    return s * (1 - s)

# ================================================================================================
# KEY NEURAL NETWORK CONCEPTS AND DEFINITIONS
# ================================================================================================

"""
FUNDAMENTAL DEFINITIONS:

1. VECTOR: A one-dimensional array of numbers (e.g., [1, 2, 3])
   - In our case: input vector x = [sex, age]
   - Represented as numpy array with shape (n,)
   - Echo: "Creating input vector with 2 features: sex and age"

2. SCALAR: A single number (not an array)
   - Examples: learning rate (0.1), bias (single value), loss (single value)
   - Represented as float or numpy scalar
   - Echo: "Working with scalar values for parameters and loss"

3. WEIGHTS (w): Parameters that determine the importance of each input feature
   - Mathematical symbol: w = [w₁, w₂, ..., wₙ]
   - Shape: (number_of_features,) - in our case (2,) for sex and age
   - Updated during training to minimize loss
   - Echo: "Initializing weight vector for feature importance learning"

4. BIAS (b): An additional parameter that shifts the decision boundary
   - Mathematical symbol: b
   - Shape: scalar value
   - Allows the model to make predictions even when all inputs are zero
   - Echo: "Adding bias term to enable flexible decision boundaries"

5. SIGMOID FUNCTION: Activation function that maps any real number to range (0,1)
   - Mathematical formula: σ(z) = 1 / (1 + e^(-z))
   - Purpose: Converts linear output to probability for binary classification
   - Output range: (0, 1) - perfect for probability interpretation
   - Echo: "Applying sigmoid activation for probability output"

6. FORWARD PROPAGATION: Process of computing predictions from inputs
   - Step 1: Linear combination: z = x·w + b (dot product + bias)
   - Step 2: Activation: y_pred = σ(z) (apply sigmoid)
   - Direction: Input → Output
   - Echo: "Executing forward pass: input → linear → activation → output"

7. BACKWARD PROPAGATION (BACKPROP): Process of computing gradients for parameter updates
   - Uses chain rule to compute: ∂Loss/∂weights and ∂Loss/∂bias
   - Direction: Output → Input (opposite of forward pass)
   - Purpose: Determine how to adjust parameters to reduce loss
   - Echo: "Computing gradients via backpropagation: output → gradients → updates"

8. MSE (Mean Squared Error): Loss function measuring prediction accuracy
   - Mathematical formula: MSE = (1/n) * Σ(y_pred - y_actual)²
   - Lower values = better predictions
   - Always positive, minimum value is 0 (perfect predictions)
   - Echo: "Calculating MSE loss to measure prediction quality"

MATHEMATICAL SYMBOLS USED:
- σ (sigma): Sigmoid function
- w: Weights vector
- b: Bias scalar
- x: Input vector
- z: Weighted sum (linear combination)
- y_pred: Predicted output
- y_actual: True/target output
- ∂: Partial derivative symbol
- ·: Dot product operation
- Σ: Summation symbol
"""

# ================================================================================================
# Phase 1: Data Loading Function
# ================================================================================================

def load_data(filename, target_col):
    """
    Load and preprocess data from CSV or TXT files
    
    VECTORS AND SCALARS IN ACTION:
    - X (output): Matrix of input vectors, each row is a sample vector [sex, age]
    - Y (output): Vector of scalar target values [0 or 1 for each sample]
    
    Args:
        filename: Path to data file
        target_col: Column name for target variable (scalar string)
    
    Returns:
        X: Input features matrix - shape (n_samples, 2) - collection of input vectors
        Y: Target variable vector - shape (n_samples, 1) - collection of scalars
        all_targets: All target columns for reference
    """
    print(f"Echo: Loading data from {filename} for target column '{target_col}'")
    
    # Handle different file formats
    if filename.endswith('.csv'):
        print("Echo: Detected CSV format, loading with pandas read_csv")
        df = pd.read_csv(filename)
    elif filename.endswith('.txt'):
        print("Echo: Detected TXT format, attempting flexible parsing")
        try:
            df = pd.read_csv(filename, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(filename, sep='\t')
    else:
        raise ValueError('Unsupported file format. Use .csv or .txt')
    
    # Extract features: sex and age as input VECTORS (each row is a vector)
    print("Echo: Extracting input features (sex, age) as feature vectors")
    X = df[['sex', 'age']].values.astype(float)  # Shape: (n_samples, 2)
    
    # Extract target variable as SCALARS (each element is a scalar 0 or 1)
    print(f"Echo: Extracting target variable '{target_col}' as scalar labels")
    Y = df[[target_col]].values.astype(float)    # Shape: (n_samples, 1)
    
    # Keep all target columns for reference
    print("Echo: Preserving all target columns for analysis reference")
    all_targets = df[['will_buy_fries', 'will_buy_hamburger']].values.astype(int)
    
    print(f"Echo: Data loading complete - X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y, all_targets

# ================================================================================================
# Phase 2: Neural Network Training Function
# ================================================================================================

def train_and_save_weights(target_column, weights_file):
    """
    Train a single-neuron neural network using manual gradient descent
    
    NEURAL NETWORK ARCHITECTURE:
    Input Layer: 2 neurons (sex, age) - INPUT VECTOR x = [x₁, x₂]
    Hidden Layer: None (single neuron network)
    Output Layer: 1 neuron with sigmoid activation - OUTPUT SCALAR y_pred
    
    MATHEMATICAL FLOW:
    z = x·w + b        (Linear combination - SCALAR result)
    y_pred = σ(z)      (Sigmoid activation - SCALAR result)
    
    Args:
        target_column: Name of target column to predict (scalar string)
        weights_file: File to save trained weights (scalar string)
    """
    print(f"Echo: Starting neural network training for target '{target_column}'")
    
    # Load data for training
    try:
        print("Echo: Attempting to load training data...")
        X, Y, all_targets = load_data('data.csv', target_column)
        print(f"Echo: Successfully loaded {len(X)} training samples")
    except Exception as e:
        print(f"Echo: ERROR - Failed to load data: {e}")
        sys.exit(1)

    # ============================================================================================
    # PARAMETER INITIALIZATION
    # ============================================================================================
    
    print("Echo: Initializing neural network parameters...")
    
    # WEIGHTS DEFINITION AND INITIALIZATION:
    # w is a VECTOR of parameters that determine feature importance
    # Shape: (2,) for our two features [weight_for_sex, weight_for_age]
    # Mathematical symbol: w = [w₁, w₂]
    w = np.random.randn(2)  # Random initialization from normal distribution
    print(f"Echo: Initialized random weights (VECTOR): {w}")
    print(f"Echo:   w₁ (sex weight): {w[0]:.4f}")
    print(f"Echo:   w₂ (age weight): {w[1]:.4f}")
    
    # BIAS DEFINITION AND INITIALIZATION:
    # b is a SCALAR parameter that shifts the decision boundary
    # Allows model to make predictions even when all inputs are zero
    # Mathematical symbol: b
    b = np.random.randn(1)[0]  # Random scalar initialization
    print(f"Echo: Initialized random bias (SCALAR): {b:.4f}")

    # HYPERPARAMETERS (all SCALARS):
    learning_rate = 0.1  # Step size for gradient descent (how big steps to take)
    num_epochs = 1000    # Number of complete passes through the dataset
    losses = []          # List to track MSE loss over time (will contain scalars)

    print(f"Echo: Set training hyperparameters:")
    print(f"Echo:   Learning rate: {learning_rate} (SCALAR)")
    print(f"Echo:   Number of epochs: {num_epochs} (SCALAR)")

    # ============================================================================================
    # TRAINING LOOP - FORWARD AND BACKWARD PROPAGATION
    # ============================================================================================
    
    print("Echo: Beginning training loop with forward and backward propagation...")
    
    for epoch in range(num_epochs):
        
        # ========================================================================================
        # FORWARD PROPAGATION
        # ========================================================================================
        
        # Step 1: LINEAR COMBINATION (z = X·w + b)
        # Compute weighted sum for each sample
        if epoch == 0:
            print("Echo: Executing forward propagation step 1 - linear combination")
        z = np.dot(X, w) + b  # Matrix-vector dot product + scalar bias
        if epoch == 0:
            print(f"Echo: Computed weighted sums z with shape: {z.shape} (VECTOR)")
        
        # Step 2: ACTIVATION (y_pred = σ(z))
        # Apply sigmoid to convert to probabilities
        if epoch == 0:
            print("Echo: Executing forward propagation step 2 - sigmoid activation")
        y_pred = sigmoid(z)   # Element-wise sigmoid application
        if epoch == 0:
            print(f"Echo: Generated predictions y_pred with shape: {y_pred.shape} (VECTOR of probabilities)")

        # ========================================================================================
        # MSE LOSS CALCULATION
        # ========================================================================================
        
        if epoch == 0:
            print("Echo: Calculating Mean Squared Error loss...")
        loss = np.mean((y_pred.reshape(-1, 1) - Y) ** 2)  # MSE computation
        losses.append(loss)  # Store scalar loss value
        
        if epoch == 0:
            print(f"Echo: MSE loss calculation details:")
            print(f"Echo:   Predictions shape: {y_pred.reshape(-1, 1).shape}")
            print(f"Echo:   Targets shape: {Y.shape}")
            print(f"Echo:   Current MSE (SCALAR): {loss:.6f}")

        # ========================================================================================
        # BACKWARD PROPAGATION
        # ========================================================================================
        
        if epoch == 0:
            print("Echo: Beginning backward propagation (gradient computation)...")
        
        # Step 1: ∂Loss/∂y_pred (gradient of MSE loss w.r.t. predictions)
        # For MSE: ∂/∂y_pred[(y_pred - y_actual)²] = 2(y_pred - y_actual)
        if epoch == 0:
            print("Echo: Computing ∂Loss/∂y_pred (loss gradient w.r.t. predictions)")
        dloss_dy = 2 * (y_pred.reshape(-1, 1) - Y) / Y.size  # Shape: (n_samples, 1)
        
        # Step 2: ∂y_pred/∂z (sigmoid derivative)
        # How much sigmoid output changes for small changes in weighted sum
        if epoch == 0:
            print("Echo: Computing ∂y_pred/∂z (sigmoid derivative)")
        dy_dz = sigmoid_derivative(z).reshape(-1, 1)  # Shape: (n_samples, 1)
        
        # Step 3: Chain rule - ∂Loss/∂z
        # Combines steps 1 and 2 using chain rule
        if epoch == 0:
            print("Echo: Applying chain rule to compute ∂Loss/∂z")
        dloss_dz = dloss_dy * dy_dz  # Element-wise multiplication, shape: (n_samples, 1)
        
        # Step 4: ∂Loss/∂w (gradient w.r.t. weights)
        # For z = X·w + b, ∂z/∂w = X (input features)
        if epoch == 0:
            print("Echo: Computing ∂Loss/∂w (gradient w.r.t. weights)")
        dloss_dw = np.dot(X.T, dloss_dz).flatten()  # Shape: (2,) - VECTOR gradient
        
        # Step 5: ∂Loss/∂b (gradient w.r.t. bias)
        # For z = X·w + b, ∂z/∂b = 1
        if epoch == 0:
            print("Echo: Computing ∂Loss/∂b (gradient w.r.t. bias)")
        dloss_db = np.sum(dloss_dz)  # Shape: scalar - SCALAR gradient

        if epoch == 0:
            print(f"Echo: Backward propagation gradient summary:")
            print(f"Echo:   dloss_dw (∂L/∂w) shape: {dloss_dw.shape} (VECTOR)")
            print(f"Echo:   dloss_db (∂L/∂b) type: scalar")
            print(f"Echo:   Weight gradients: {dloss_dw}")
            print(f"Echo:   Bias gradient: {dloss_db:.6f}")

        # ========================================================================================
        # PARAMETER UPDATES - GRADIENT DESCENT
        # ========================================================================================
        
        if epoch == 0:
            print("Echo: Updating parameters using gradient descent...")
        
        # Update WEIGHTS (vector update)
        if epoch == 0:
            print(f"Echo: Updating weights: w_new = w_old - {learning_rate} * gradient")
        w -= learning_rate * dloss_dw  # Vector subtraction
        
        # Update BIAS (scalar update)  
        if epoch == 0:
            print(f"Echo: Updating bias: b_new = b_old - {learning_rate} * gradient")
        b -= learning_rate * dloss_db  # Scalar subtraction

        # Progress reporting every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Echo: [{target_column}] Epoch {epoch+1}, Loss: {loss:.4f}")

    # Training completed - display final parameters
    print(f"Echo: Training completed for {target_column}!")
    print(f"Echo: Final trained weights (VECTOR): {w}")
    print(f"Echo:   w₁ (sex weight): {w[0]:.4f}")
    print(f"Echo:   w₂ (age weight): {w[1]:.4f}")
    print(f"Echo: Final trained bias (SCALAR): {b:.4f}")

    # ============================================================================================
    # Save Training Results
    # ============================================================================================
    
    print("Echo: Saving training results to files...")
    
    # Save weights to file (append mode for multiple runs)
    with open(weights_file, 'a') as f:
        np.savetxt(f, w.reshape(1, -1))
    print(f'Echo: Weights successfully appended to {weights_file}')

    # Save loss curve for analysis
    loss_df = pd.DataFrame({'loss': losses})
    loss_df.to_csv(f'loss_curve_{target_column}.csv', index=False)
    print(f'Echo: Loss curve successfully saved to loss_curve_{target_column}.csv')

# ================================================================================================
# Phase 3: Model Execution - Train Two Separate Models
# ================================================================================================

print("Echo: === TRAINING PHASE INITIATED ===")
print("Echo: Training separate models for binary classification tasks")

# Train model for predicting fries purchases
print("Echo: Starting Training Model 1: Fries Prediction")
train_and_save_weights('will_buy_fries', 'weights_fries.txt')

print("Echo: " + "="*80)

# Train model for predicting hamburger purchases  
print("Echo: Starting Training Model 2: Hamburger Prediction")
train_and_save_weights('will_buy_hamburger', 'weights_hamburger.txt')

# ================================================================================================
# Phase 4: Prediction and Detailed Analysis
# ================================================================================================

print("Echo: === PREDICTION AND ANALYSIS PHASE INITIATED ===")
print("Echo: Demonstrating forward propagation with trained models")

# Detailed Prediction Loop for Both Models
for target_col in ['will_buy_fries', 'will_buy_hamburger']:
    print(f"Echo: {'='*60}")
    print(f"Echo: BEGINNING ANALYSIS FOR: {target_col.upper()}")
    print(f"Echo: {'='*60}")
    
    # Load trained model parameters
    print("Echo: Loading trained model parameters...")
    X, Y, all_targets = load_data('data.csv', target_col)
    weights_file = 'weights_fries.txt' if target_col == 'will_buy_fries' else 'weights_hamburger.txt'
    w = np.loadtxt(weights_file)[-1]  # Load most recent weights (VECTOR)
    b = 0  # Bias set to 0 (SCALAR) - limitation of this implementation
    
    print(f"Echo: Successfully loaded weights (VECTOR): {w}")
    print(f"Echo: Using bias (SCALAR): {b}")
    
    results = []  # Store results for CSV export
    
    # Sample-by-Sample Analysis with Detailed Mathematical Breakdown
    print("Echo: Beginning sample-by-sample analysis with detailed breakdown...")
    
    for i, x in enumerate(X):
        print(f"Echo: --- ANALYZING SAMPLE {i+1} ---")
        
        # INPUT VECTOR ANALYSIS
        sex_str = 'male' if x[0] == 0 else 'female'
        print(f"Echo: Processing input vector x: {x} (sex={sex_str}, age={x[1]})")
        print(f"Echo: Input x is a VECTOR with shape: {x.shape}")
        
        # FORWARD PROPAGATION STEP BY STEP
        print(f"Echo: Executing forward propagation for sample {i+1}...")
        
        # Step 1: Linear combination (dot product + bias)
        print("Echo: Step 1 - Computing linear combination (dot product + bias)")
        dot_product = np.dot(x, w)  # SCALAR result from vector dot product
        z = dot_product + b         # SCALAR weighted sum
        print(f"Echo: Linear combination calculation:")
        print(f"Echo:   z = x·w + b")
        print(f"Echo:   z = {x} · {w} + {b}")
        print(f"Echo:   z = {dot_product:.4f} + {b} = {z:.4f} (SCALAR)")
        
        # Step 2: Sigmoid activation
        print("Echo: Step 2 - Applying sigmoid activation function")
        pred = sigmoid(z)  # SCALAR probability
        print(f"Echo: Sigmoid activation calculation:")
        print(f"Echo:   y_pred = σ(z) = σ({z:.4f})")
        print(f"Echo:   y_pred = {pred:.4f} (SCALAR probability)")
        
        # ACTUAL VS PREDICTED
        fries, hamburger = all_targets[i]
        if fries == 1:
            actual_str = 'fries'
        elif hamburger == 1:
            actual_str = 'hamburger'
        else:
            actual_str = 'none'
        
        print(f"Echo: Prediction comparison:")
        print(f"Echo:   Predicted probability: {pred:.4f}")
        print(f"Echo:   Actual purchase: {actual_str} (numeric: {int(Y[i][0])})")
        
        # GRADIENT ANALYSIS (Educational - shows backprop calculations)
        print(f"Echo: Performing educational gradient analysis for sample {i+1}...")
        
        # MSE gradients for this single sample
        dloss_dy = 2 * (pred - Y[i][0])  # SCALAR gradient
        dy_dz = pred * (1 - pred)        # SCALAR sigmoid derivative  
        dloss_dz = dloss_dy * dy_dz      # SCALAR chain rule result
        dloss_dw = x * dloss_dz          # VECTOR gradient (broadcasting)
        dloss_db = dloss_dz              # SCALAR gradient
        
        print(f"Echo: Gradient calculations for sample {i+1}:")
        print(f"Echo:   ∂Loss/∂y_pred: {dloss_dy:.6f} (SCALAR)")
        print(f"Echo:   ∂y_pred/∂z:    {dy_dz:.6f} (SCALAR)")  
        print(f"Echo:   ∂Loss/∂z:      {dloss_dz:.6f} (SCALAR)")
        print(f"Echo:   ∂Loss/∂w:      {dloss_dw} (VECTOR)")
        print(f"Echo:   ∂Loss/∂b:      {dloss_db:.6f} (SCALAR)")
        
        # Store results for CSV export
        results.append({
            'sex': sex_str,
            'age': x[1],
            'predicted': pred,
            'actual': actual_str,
            'actual_num': int(Y[i][0])
        })
    
    # Export Results
    print("Echo: Exporting analysis results to CSV file...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'predictions_{target_col}.csv', index=False)
    print(f'Echo: Results successfully exported to predictions_{target_col}.csv')

print(f"Echo: {'='*80}")
print("Echo: COMPLETE ANALYSIS FINISHED SUCCESSFULLY")
print(f"Echo: {'='*80}")

# ================================================================================================
# SUMMARY OF KEY CONCEPTS DEMONSTRATED:
# ================================================================================================

"""
Echo: CONCEPTS DEMONSTRATED IN THIS PROGRAM:

Echo: 1. VECTORS vs SCALARS:
Echo:    - Input features: VECTORS (e.g., [sex, age])
Echo:    - Weights: VECTOR of parameters
Echo:    - Bias: SCALAR parameter
Echo:    - Predictions: SCALARS (probabilities)
Echo:    - Loss: SCALAR value

Echo: 2. FORWARD PROPAGATION:
Echo:    - Linear combination: z = x·w + b
Echo:    - Activation: y_pred = σ(z)
Echo:    - Data flows from input to output

Echo: 3. BACKWARD PROPAGATION:
Echo:    - Compute gradients using chain rule
Echo:    - Update parameters to minimize loss
Echo:    - Data flows from output to input

Echo: 4. MATHEMATICAL OPERATIONS:
Echo:    - Dot product: vector × vector → scalar
Echo:    - Broadcasting: scalar operations with vectors
Echo:    - Element-wise operations: apply function to each element

Echo: 5. NEURAL NETWORK COMPONENTS:
Echo:    - Weights: learnable parameters that determine feature importance
Echo:    - Bias: learnable parameter that shifts decision boundary
Echo:    - Sigmoid: activation function for binary classification
Echo:    - MSE: loss function measuring prediction quality

Echo: This implementation shows the fundamental building blocks of neural networks
Echo: with explicit mathematical operations and detailed educational explanations.
""" 
