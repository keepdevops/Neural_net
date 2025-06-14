import numpy as np
import pandas as pd
import duckdb
import os
from datetime import datetime, timedelta

def extract_stock_features_and_labels(db_path, 
                                    features_csv="stock_features.csv", 
                                    labels_csv="stock_labels.csv",
                                    years_back=2):
    """
    Extract stock features (open, high, low, vol) and labels (close) from DuckDB
    Save as separate CSV files for X and y
    
    Parameters:
    db_path: path to the DuckDB file
    features_csv: path to save features CSV (X)
    labels_csv: path to save labels CSV (y)
    years_back: number of years to look back
    
    Returns:
    X, y: numpy arrays, CSV file paths
    """
    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Calculate date range (last N years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        print(f"Extracting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Check available tables
        tables_result = conn.execute("SHOW TABLES").fetchall()
        tables = [table[0] for table in tables_result]
        print(f"Available tables: {tables}")
        
        # Find the stock data table
        stock_table = None
        for table in tables:
            if any(keyword in table.lower() for keyword in ['stock', 'ticker', 'data', 'ohlc', 'price']):
                stock_table = table
                break
        
        if not stock_table:
            stock_table = tables[0]
        
        print(f"Using table: {stock_table}")
        
        # Get table schema
        columns_info = conn.execute(f"DESCRIBE {stock_table}").fetchall()
        available_columns = [col[0] for col in columns_info]
        print(f"Available columns: {available_columns}")
        
        # Map required columns to actual column names
        required_columns = ['open', 'high', 'low', 'vol', 'close']
        column_mapping = {}
        
        for required in required_columns:
            for available in available_columns:
                if required.lower() == available.lower():
                    column_mapping[required] = available
                    break
                # Handle volume variations
                elif required == 'vol' and any(vol_var in available.lower() for vol_var in ['volume', 'vol']):
                    column_mapping[required] = available
                    break
        
        print(f"Column mapping: {column_mapping}")
        
        # Check if we have all required columns
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns}")
        
        # Find timestamp column
        timestamp_column = None
        for col in available_columns:
            if any(time_keyword in col.lower() for time_keyword in ['timestamp', 'date', 'time', 'datetime']):
                timestamp_column = col
                break
        
        if timestamp_column:
            print(f"Using timestamp column: {timestamp_column}")
            date_filter = f"WHERE {timestamp_column} >= '{start_date.strftime('%Y-%m-%d')}' AND {timestamp_column} <= '{end_date.strftime('%Y-%m-%d')}'"
            order_by = f"ORDER BY {timestamp_column}"
        else:
            print("Warning: No timestamp column found. Extracting all data.")
            date_filter = ""
            order_by = ""
        
        # Build query to get all required data
        all_columns_sql = [column_mapping[col] for col in required_columns if col in column_mapping]
        if timestamp_column:
            all_columns_sql.append(timestamp_column)
        
        columns_sql = ', '.join(all_columns_sql)
        
        query = f"""
        SELECT {columns_sql}
        FROM {stock_table}
        {date_filter}
        {order_by}
        """
        
        print(f"Executing query: {query}")
        
        # Execute query
        result = conn.execute(query).fetchall()
        conn.close()
        
        if not result:
            raise ValueError("No data returned from query")
        
        print(f"Retrieved {len(result)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(result, columns=all_columns_sql)
        
        # Separate features and labels
        feature_columns = ['open', 'high', 'low', 'vol']
        available_feature_cols = [column_mapping[col] for col in feature_columns if col in column_mapping]
        
        # Create features DataFrame (X)
        df_features = df[available_feature_cols].copy()
        # Rename to standard names
        feature_rename_map = {column_mapping[col]: col for col in feature_columns if col in column_mapping}
        df_features = df_features.rename(columns=feature_rename_map)
        
        # Create labels DataFrame (y) - just the close prices
        if 'close' in column_mapping:
            df_labels = pd.DataFrame({
                'close': df[column_mapping['close']]
            })
        else:
            raise ValueError("Close column not found - required for labels")
        
        # Add timestamp to both if available
        if timestamp_column:
            df_features['timestamp'] = df[timestamp_column]
            df_labels['timestamp'] = df[timestamp_column]
        
        # Save features to CSV
        df_features.to_csv(features_csv, index=False, float_format='%.6f')
        print(f"Features saved to: {features_csv}")
        print(f"Feature columns: {list(df_features.columns)}")
        
        # Save labels to CSV
        df_labels.to_csv(labels_csv, index=False, float_format='%.6f')
        print(f"Labels saved to: {labels_csv}")
        print(f"Label columns: {list(df_labels.columns)}")
        
        # Convert to numpy arrays
        feature_cols_for_array = [col for col in df_features.columns if col != 'timestamp']
        X = df_features[feature_cols_for_array].values.astype(np.float32)
        y = df_labels['close'].values.astype(np.float32).reshape(-1, 1)
        
        print(f"\nNumpy arrays created:")
        print(f"X shape: {X.shape} (features: {feature_cols_for_array})")
        print(f"y shape: {y.shape} (close prices)")
        
        # Display sample data
        print(f"\nSample data:")
        print("Features (X):")
        print(df_features.head())
        print("\nLabels (y):")
        print(df_labels.head())
        
        return X, y, features_csv, labels_csv
        
    except Exception as e:
        print(f"Error extracting stock data: {e}")
        return None, None, None, None

def create_binary_labels_from_close(y_close):
    """
    Convert close prices to binary labels for classification
    
    Parameters:
    y_close: array of close prices
    
    Returns:
    y_binary: binary labels (1 if price goes up next day, 0 if down)
    """
    try:
        # Create binary labels: 1 if next day price is higher, 0 if lower
        y_binary = np.zeros_like(y_close)
        
        for i in range(len(y_close) - 1):
            if y_close[i + 1] > y_close[i]:
                y_binary[i] = 1
            else:
                y_binary[i] = 0
        
        # Last day gets same label as previous day
        if len(y_close) > 1:
            y_binary[-1] = y_binary[-2]
        
        print(f"Created binary labels from close prices:")
        print(f"  Up days (1): {np.sum(y_binary)}")
        print(f"  Down days (0): {len(y_binary) - np.sum(y_binary)}")
        
        return y_binary.astype(np.float32)
        
    except Exception as e:
        print(f"Error creating binary labels: {e}")
        return None

def extract_tsla_training_data(db_path="/Users/porupine/redline/data/tsla.us_data.duckdb",
                             features_csv="tsla_features.csv",
                             labels_csv="tsla_labels.csv",
                             binary_labels=True,
                             years_back=2):
    """
    Complete pipeline for TSLA data extraction
    
    Parameters:
    db_path: path to TSLA DuckDB file
    features_csv: output path for features
    labels_csv: output path for labels
    binary_labels: if True, convert close prices to binary up/down labels
    years_back: years of historical data
    
    Returns:
    X, y: training data ready for neural network
    """
    print("=== TSLA Training Data Extraction ===")
    print(f"Source: {db_path}")
    print(f"Features output: {features_csv}")
    print(f"Labels output: {labels_csv}")
    print(f"Time range: Last {years_back} years")
    print(f"Binary labels: {binary_labels}")
    
    # Extract data
    X, y_close, feat_csv, lab_csv = extract_stock_features_and_labels(
        db_path, features_csv, labels_csv, years_back
    )
    
    if X is None:
        return None, None
    
    # Convert to binary labels if requested
    if binary_labels:
        y = create_binary_labels_from_close(y_close)
        
        # Save binary labels to separate CSV
        binary_labels_csv = labels_csv.replace('.csv', '_binary.csv')
        df_binary = pd.DataFrame({'binary_label': y.flatten()})
        df_binary.to_csv(binary_labels_csv, index=False)
        print(f"Binary labels saved to: {binary_labels_csv}")
    else:
        y = y_close
    
    print(f"\n=== Extraction Complete ===")
    print(f"Features (X): {X.shape}")
    print(f"Labels (y): {y.shape}")
    print(f"Ready for neural network training!")
    
    return X, y

def load_features_and_labels_csv(features_csv, labels_csv):
    """
    Load features and labels from separate CSV files
    
    Parameters:
    features_csv: path to features CSV file
    labels_csv: path to labels CSV file
    
    Returns:
    X, y: numpy arrays
    """
    try:
        # Load features
        df_features = pd.read_csv(features_csv)
        feature_cols = [col for col in df_features.columns if col != 'timestamp']
        X = df_features[feature_cols].values.astype(np.float32)
        
        # Load labels
        df_labels = pd.read_csv(labels_csv)
        y = df_labels['close'].values.astype(np.float32).reshape(-1, 1)
        
        print(f"Loaded from CSV:")
        print(f"  Features: {features_csv} - shape {X.shape}")
        print(f"  Labels: {labels_csv} - shape {y.shape}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None, None

def save_arrays_to_csv(X, y, features_csv="features.csv", labels_csv="labels.csv"):
    """
    Save numpy arrays directly to CSV files
    
    Parameters:
    X: features array
    y: labels array
    features_csv: output path for features
    labels_csv: output path for labels
    """
    try:
        # Save features
        feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
        df_features = pd.DataFrame(X, columns=feature_names)
        df_features.to_csv(features_csv, index=False, float_format='%.6f')
        
        # Save labels
        if len(y.shape) == 1:
            df_labels = pd.DataFrame({'label': y})
        else:
            df_labels = pd.DataFrame({'label': y.flatten()})
        df_labels.to_csv(labels_csv, index=False, float_format='%.6f')
        
        print(f"Arrays saved to CSV:")
        print(f"  Features: {features_csv} - {X.shape}")
        print(f"  Labels: {labels_csv} - {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error saving arrays to CSV: {e}")
        return False 