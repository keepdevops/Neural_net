import numpy as np
import duckdb
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime, timedelta
import random
from duckdb_loader import (
    load_data_from_duckdb_enhanced,
    launch_enhanced_gui_loader,
    create_sample_duckdb,
    find_duckdb_files
)
from duckdb_to_csv_converter import (
    extract_tsla_training_data,
    load_features_and_labels_csv,
    save_arrays_to_csv
)
#

def find_duckdb_files(directory_path):
    """
    Find all DuckDB files in a directory
    
    Parameters:
    directory_path: path to search for DuckDB files
    
    Returns:
    list of DuckDB file paths
    """
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return []
    
    # Common DuckDB file extensions
    duckdb_patterns = [
        os.path.join(directory_path, "*.db"),
        os.path.join(directory_path, "*.duckdb"),
        os.path.join(directory_path, "*.duck"),
        os.path.join(directory_path, "*.ddb")
    ]
    
    duckdb_files = []
    for pattern in duckdb_patterns:
        duckdb_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    duckdb_files = sorted(list(set(duckdb_files)))
    
    print(f"Found {len(duckdb_files)} DuckDB files in {directory_path}:")
    for i, file_path in enumerate(duckdb_files):
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  {i+1}. {os.path.basename(file_path)} ({file_size:.1f} KB)")
    
    return duckdb_files

def select_duckdb_file(directory_path=None, file_path=None, auto_select=True):
    """
    Select a DuckDB file from directory or use specified file path
    
    Parameters:
    directory_path: directory to search for DuckDB files
    file_path: specific file path (overrides directory_path)
    auto_select: if True, automatically select first file found
    
    Returns:
    selected file path or None
    """
    if file_path:
        # Use specific file path
        if os.path.exists(file_path):
            print(f"Using specified DuckDB file: {file_path}")
            return file_path
        else:
            print(f"Specified file does not exist: {file_path}")
            return None
    
    if directory_path:
        # Search directory for DuckDB files
        duckdb_files = find_duckdb_files(directory_path)
        
        if not duckdb_files:
            print(f"No DuckDB files found in directory: {directory_path}")
            return None
        
        if auto_select or len(duckdb_files) == 1:
            # Auto-select first file or if only one file
            selected_file = duckdb_files[0]
            print(f"Auto-selected: {os.path.basename(selected_file)}")
            return selected_file
        else:
            # Interactive selection (for manual mode)
            print("\nMultiple DuckDB files found. Please select one:")
            for i, file_path in enumerate(duckdb_files):
                print(f"  {i+1}. {os.path.basename(file_path)}")
            
            try:
                choice = int(input("Enter your choice (number): ")) - 1
                if 0 <= choice < len(duckdb_files):
                    return duckdb_files[choice]
                else:
                    print("Invalid choice")
                    return None
            except (ValueError, KeyboardInterrupt):
                print("Selection cancelled")
                return None
    
    return None

def get_column_types(conn, table_name):
    """
    Get column types from DuckDB table
    
    Parameters:
    conn: DuckDB connection
    table_name: name of the table
    
    Returns:
    dict: column_name -> data_type mapping
    """
    try:
        # Get column information with types
        columns_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_types = {}
        
        for col_info in columns_info:
            col_name = col_info[0]
            col_type = col_info[1].upper()
            
            # Determine if column is numeric
            is_numeric = any(numeric_type in col_type for numeric_type in [
                'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT',
                'DOUBLE', 'FLOAT', 'REAL', 'DECIMAL', 'NUMERIC'
            ])
            
            column_types[col_name] = {
                'type': col_type,
                'is_numeric': is_numeric
            }
        
        return column_types
        
    except Exception as e:
        print(f"Error getting column types: {e}")
        return {}

def load_data_from_duckdb_enhanced(db_path=None, directory_path=None, table_name=None, feature_columns=None, label_column=None, query=None, auto_select=True):
    """
    Enhanced data loading from DuckDB with proper type handling
    
    Parameters:
    db_path: specific path to the DuckDB file
    directory_path: directory to search for DuckDB files (used if db_path is None)
    table_name: name of the table containing the data (optional if using custom query)
    feature_columns: list of column names for features (X) - if None, auto-detect numeric columns
    label_column: column name for labels (y) - if None, use last numeric column
    query: custom SQL query (overrides table_name and column specifications)
    auto_select: automatically select first DuckDB file found in directory
    
    Returns:
    X, y: numpy arrays for features and labels
    """
    try:
        # Select DuckDB file
        selected_db_path = select_duckdb_file(directory_path, db_path, auto_select)
        
        if not selected_db_path:
            raise ValueError("No valid DuckDB file found or selected")
        
        # Connect to DuckDB
        conn = duckdb.connect(selected_db_path)
        
        if query:
            # Use custom query if provided
            print(f"Executing custom query: {query}")
            result = conn.execute(query).fetchall()
            
            # For custom queries, we need to get column info differently
            # Execute query with LIMIT 0 to get column names and types
            describe_result = conn.execute(f"DESCRIBE ({query})").fetchall()
            all_columns = [col[0] for col in describe_result]
            
            # Assume all columns in custom query are intended to be numeric
            if feature_columns is None:
                feature_columns = all_columns[:-1]
            if label_column is None:
                label_column = all_columns[-1]
                
        else:
            # Auto-detect table if not specified
            if table_name is None:
                tables = conn.execute("SHOW TABLES").fetchall()
                if not tables:
                    raise ValueError("No tables found in database")
                table_name = tables[0][0]  # Use first table
                print(f"Auto-detected table: {table_name}")
            
            # Get column types
            column_types = get_column_types(conn, table_name)
            all_columns = list(column_types.keys())
            numeric_columns = [col for col, info in column_types.items() if info['is_numeric']]
            non_numeric_columns = [col for col, info in column_types.items() if not info['is_numeric']]
            
            print(f"Numeric columns: {numeric_columns}")
            print(f"Non-numeric columns: {non_numeric_columns}")
            
            # Auto-detect columns if not specified, using only numeric columns
            if feature_columns is None:
                # Exclude common non-feature columns
                excluded_cols = ['ticker', 'symbol', 'timestamp', 'date', 'time', 'id', 'format']
                feature_columns = [col for col in numeric_columns 
                                 if col.lower() not in [exc.lower() for exc in excluded_cols]]
                
                # If we still have too many columns, exclude the last one for label
                if len(feature_columns) > 1:
                    # Remove the last column to use as label
                    if label_column is None:
                        potential_labels = [col for col in feature_columns if col.lower() in ['close', 'target', 'label', 'y']]
                        if potential_labels:
                            label_column = potential_labels[0]
                            feature_columns = [col for col in feature_columns if col != label_column]
                        else:
                            label_column = feature_columns[-1]
                            feature_columns = feature_columns[:-1]
            
            if label_column is None:
                # Find a suitable label column
                potential_labels = [col for col in numeric_columns if col.lower() in ['close', 'target', 'label', 'y']]
                if potential_labels:
                    label_column = potential_labels[0]
                else:
                    label_column = numeric_columns[-1]
                    
            print(f"Auto-detected features: {feature_columns}")
            print(f"Auto-detected label: {label_column}")
            
            # Build and execute query with only numeric columns
            feature_cols = ', '.join(feature_columns)
            query = f"SELECT {feature_cols}, {label_column} FROM {table_name}"
            print(f"Executing query: {query}")
            result = conn.execute(query).fetchall()
        
        # Convert to numpy arrays (now all data should be numeric)
        if not result:
            raise ValueError("No data returned from query")
            
        # Convert to numpy array - all columns should now be numeric
        try:
            data = np.array(result, dtype=np.float32)
        except ValueError as e:
            print(f"Error converting to float: {e}")
            print("Sample data:")
            for i, row in enumerate(result[:3]):  # Show first 3 rows
                print(f"  Row {i}: {row}")
            raise ValueError("Data contains non-numeric values. Please check your column selection.")
            
        X = data[:, :-1]  # All columns except last (features)
        y = data[:, -1:] # Last column (labels)
        
        conn.close()
        
        print(f"Successfully loaded {len(X)} samples from DuckDB")
        print(f"Database: {os.path.basename(selected_db_path)}")
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Feature ranges: min={X.min(axis=0)}, max={X.max(axis=0)}")
        
        # Handle label distribution safely
        try:
            unique_labels, counts = np.unique(y.astype(int).flatten(), return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"Label distribution: {label_dist}")
        except:
            print(f"Label range: min={y.min():.3f}, max={y.max():.3f}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data from DuckDB: {e}")
        if 'conn' in locals():
            conn.close()
        return None, None

def create_sample_duckdb(db_path="sample_data.db"):
    """
    Create a sample DuckDB file with training data for testing
    """
    try:
        conn = duckdb.connect(db_path)
        
        # Create sample table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                feature1 FLOAT,
                feature2 FLOAT,
                feature3 FLOAT,
                feature4 FLOAT,
                label INTEGER
            )
        """)
        
        # Insert sample data (using the hardcoded data as example)
        sample_data = [
            (1, 5, 60, 2, 0),
            (3, 6, 75, 3, 0),
            (5, 7, 95, 4, 1),
            (0, 4, 50, 1, 0),
            (5, 8, 90, 4, 1),
            (2, 6, 75, 2, 0),
            (6, 7, 87, 3, 1),
            (5, 8, 45, 4, 0),
            (1, 1, 0, 4, 1),
            (5, 9, 91, 4, 1)
        ]
        
        conn.execute("DELETE FROM training_data")  # Clear existing data
        conn.executemany(
            "INSERT INTO training_data VALUES (?, ?, ?, ?, ?)",
            sample_data
        )
        
        conn.close()
        print(f"Sample DuckDB file created: {db_path}")
        return True
        
    except Exception as e:
        print(f"Error creating sample DuckDB: {e}")
        return False

def create_stock_data_duckdb(db_path="stock_data.db", num_records=100):
    """
    Create a DuckDB table with stock market data
    
    Parameters:
    db_path: path for the DuckDB file
    num_records: number of sample records to generate
    
    Returns:
    True if successful, False otherwise
    """
    try:
        conn = duckdb.connect(db_path)
        
        # Create stock data table
        conn.execute("""
            DROP TABLE IF EXISTS stock_data
        """)
        
        conn.execute("""
            CREATE TABLE stock_data (
                ticker VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                vol INTEGER
            )
        """)
        
        # Generate sample stock data
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD']
        base_date = datetime(2024, 1, 1)
        
        sample_data = []
        
        for i in range(num_records):
            ticker = random.choice(tickers)
            timestamp = base_date + timedelta(days=i % 365, hours=random.randint(9, 16), minutes=random.randint(0, 59))
            
            # Generate realistic stock prices
            base_price = random.uniform(50, 300)
            open_price = base_price + random.uniform(-5, 5)
            close_price = open_price + random.uniform(-10, 10)
            high_price = max(open_price, close_price) + random.uniform(0, 5)
            low_price = min(open_price, close_price) - random.uniform(0, 5)
            volume = random.randint(1000000, 50000000)
            
            sample_data.append((ticker, timestamp, open_price, high_price, low_price, close_price, volume))
        
        # Insert data
        conn.executemany("""
            INSERT INTO stock_data (ticker, timestamp, open, high, low, close, vol)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, sample_data)
        
        # Create additional tables for neural network training
        conn.execute("""
            CREATE TABLE training_features AS
            SELECT 
                open,
                high,
                low,
                vol,
                CASE WHEN close > open THEN 1 ELSE 0 END as label
            FROM stock_data
        """)
        
        conn.close()
        print(f"Stock data DuckDB file created: {db_path}")
        print(f"Generated {num_records} stock records")
        print("Tables created: stock_data, training_features")
        return True
        
    except Exception as e:
        print(f"Error creating stock data DuckDB: {e}")
        return False

def create_numpy_array_duckdb(X_array, db_path="numpy_data.db", table_name="stock_data"):
    """
    Create a DuckDB table from a numpy array with stock data columns
    
    Parameters:
    X_array: numpy array with data
    db_path: path for the DuckDB file
    table_name: name of the table to create
    
    Returns:
    True if successful, False otherwise
    """
    try:
        if X_array is None or len(X_array) == 0:
            print("No data provided to create DuckDB table")
            return False
            
        conn = duckdb.connect(db_path)
        
        # Drop existing table
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table with stock data structure
        conn.execute(f"""
            CREATE TABLE {table_name} (
                ticker VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                vol INTEGER
            )
        """)
        
        # Generate sample tickers and timestamps
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD']
        base_date = datetime(2024, 1, 1)
        
        # Convert numpy array to list of tuples with proper stock data format
        sample_data = []
        for i, row in enumerate(X_array):
            ticker = tickers[i % len(tickers)]
            timestamp = base_date + timedelta(days=i, hours=random.randint(9, 16))
            
            if len(row) >= 4:
                open_price = float(row[0]) if row[0] != 0 else random.uniform(50, 300)
                high_price = float(row[1]) if row[1] != 0 else open_price + random.uniform(0, 10)
                low_price = float(row[2]) if row[2] != 0 else open_price - random.uniform(0, 10)
                volume = int(row[3]) if row[3] != 0 else random.randint(1000000, 50000000)
                close_price = open_price + random.uniform(-5, 5)
            else:
                # Generate random data if array doesn't have enough columns
                open_price = random.uniform(50, 300)
                high_price = open_price + random.uniform(0, 10)
                low_price = open_price - random.uniform(0, 10)
                close_price = open_price + random.uniform(-5, 5)
                volume = random.randint(1000000, 50000000)
            
            sample_data.append((ticker, timestamp, open_price, high_price, low_price, close_price, volume))
        
        # Insert data
        conn.executemany(f"""
            INSERT INTO {table_name} (ticker, timestamp, open, high, low, close, vol)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, sample_data)
        
        conn.close()
        print(f"Created DuckDB table '{table_name}' with {len(sample_data)} records")
        return True
        
    except Exception as e:
        print(f"Error creating DuckDB table from numpy array: {e}")
        return False

class EnhancedDuckDBLoaderGUI:
    """
    Enhanced GUI for loading DuckDB files with proper type handling
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced DuckDB Stock Data Loader")
        self.root.geometry("1100x800")
        
        # Data storage
        self.current_db_path = None
        self.current_data = None
        self.tables = []
        self.columns = []
        self.column_types = {}
        self.loaded_X = None
        self.loaded_y = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Create the enhanced GUI layout"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(2, weight=1)
        
        ttk.Button(file_frame, text="Browse DuckDB File", 
                  command=self.browse_file).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(file_frame, text="Browse Directory", 
                  command=self.browse_directory).grid(row=0, column=1, padx=(0, 5))
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, 
                 state="readonly").grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Data creation section
        create_frame = ttk.LabelFrame(main_frame, text="Create Sample Data", padding="5")
        create_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(create_frame, text="Create Stock Data", 
                  command=self.create_stock_data).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(create_frame, text="Create from Numpy Array", 
                  command=self.create_from_numpy).grid(row=0, column=1, padx=(0, 5))
        
        ttk.Label(create_frame, text="Records:").grid(row=0, column=2, padx=(10, 5))
        self.records_var = tk.StringVar(value="100")
        ttk.Entry(create_frame, textvariable=self.records_var, width=10).grid(row=0, column=3)
        
        # Database info section
        info_frame = ttk.LabelFrame(main_frame, text="Database Information", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(3, weight=1)
        
        # Table selection
        ttk.Label(info_frame, text="Table:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(info_frame, textvariable=self.table_var, 
                                       state="readonly", width=20)
        self.table_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.table_combo.bind('<<ComboboxSelected>>', self.on_table_selected)
        
        # Column selection
        ttk.Label(info_frame, text="Label Column:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.label_col_var = tk.StringVar()
        self.label_combo = ttk.Combobox(info_frame, textvariable=self.label_col_var, 
                                       state="readonly", width=15)
        self.label_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # Action buttons
        button_frame = ttk.Frame(info_frame)
        button_frame.grid(row=0, column=4, padx=(10, 0))
        
        ttk.Button(button_frame, text="Preview Data", 
                  command=self.preview_data).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Load Data", 
                  command=self.load_data).grid(row=0, column=1)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="5")
        preview_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Create Treeview for data display
        self.tree = ttk.Treeview(preview_frame)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Status section
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Create sample data or select a DuckDB file")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky=tk.W)
        
        # Close button
        ttk.Button(status_frame, text="Use Data & Close", 
                  command=self.use_data_and_close).grid(row=0, column=1)
    
    def create_stock_data(self):
        """Create sample stock data"""
        try:
            num_records = int(self.records_var.get())
            db_path = filedialog.asksaveasfilename(
                title="Save Stock Data As",
                defaultextension=".db",
                filetypes=[("DuckDB files", "*.db"), ("All files", "*.*")],
                initialvalue="stock_data.db"
            )
            
            if db_path:
                if create_stock_data_duckdb(db_path, num_records):
                    self.load_database(db_path)
                    messagebox.showinfo("Success", f"Created stock data with {num_records} records")
                else:
                    messagebox.showerror("Error", "Failed to create stock data")
                    
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of records")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create stock data: {str(e)}")
    
    def create_from_numpy(self):
        """Create DuckDB table from the existing numpy array"""
        try:
            # Use the global X array if available
            if 'X' in globals():
                X_array = globals()['X']
            else:
                # Create sample array if none exists
                X_array = np.array([
                    [1, 5, 60, 2],
                    [3, 6, 75, 3],
                    [5, 7, 95, 4],
                    [0, 4, 50, 1],
                    [5, 8, 90, 4],
                    [2, 6, 75, 2],
                    [6, 7, 87, 3],
                    [5, 8, 45, 4],
                    [1, 1, 0, 4],
                    [5, 9, 91, 4]
                ], dtype=np.float32)
            
            db_path = filedialog.asksaveasfilename(
                title="Save Numpy Data As",
                defaultextension=".db",
                filetypes=[("DuckDB files", "*.db"), ("All files", "*.*")],
                initialvalue="numpy_stock_data.db"
            )
            
            if db_path:
                if create_numpy_array_duckdb(X_array, db_path, "stock_data"):
                    self.load_database(db_path)
                    messagebox.showinfo("Success", f"Created DuckDB table from numpy array with {len(X_array)} records")
                else:
                    messagebox.showerror("Error", "Failed to create table from numpy array")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create from numpy array: {str(e)}")
    
    def browse_file(self):
        """Open file dialog to select DuckDB file"""
        filetypes = [
            ("DuckDB files", "*.db *.duckdb *.duck *.ddb"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select DuckDB file",
            filetypes=filetypes
        )
        
        if filename:
            self.load_database(filename)
    
    def browse_directory(self):
        """Open directory dialog and show available DuckDB files"""
        directory = filedialog.askdirectory(title="Select directory with DuckDB files")
        
        if directory:
            files = find_duckdb_files(directory)
            if files:
                # Show selection dialog if multiple files
                if len(files) == 1:
                    self.load_database(files[0])
                else:
                    self.show_file_selection_dialog(files)
            else:
                messagebox.showwarning("No Files", f"No DuckDB files found in {directory}")
    
    def show_file_selection_dialog(self, files):
        """Show dialog to select from multiple DuckDB files"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select DuckDB File")
        dialog.geometry("500x300")
        dialog.grab_set()
        
        ttk.Label(dialog, text="Multiple DuckDB files found. Please select one:").pack(pady=10)
        
        # Create listbox with files
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        listbox = tk.Listbox(listbox_frame)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        for file_path in files:
            file_size = os.path.getsize(file_path) / 1024
            display_text = f"{os.path.basename(file_path)} ({file_size:.1f} KB)"
            listbox.insert(tk.END, display_text)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def select_file():
            selection = listbox.curselection()
            if selection:
                selected_file = files[selection[0]]
                dialog.destroy()
                self.load_database(selected_file)
            else:
                messagebox.showwarning("No Selection", "Please select a file")
        
        ttk.Button(button_frame, text="Select", command=select_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def load_database(self, db_path):
        """Load database and populate table list"""
        try:
            self.current_db_path = db_path
            self.file_path_var.set(db_path)
            
            # Connect and get tables
            conn = duckdb.connect(db_path)
            tables_result = conn.execute("SHOW TABLES").fetchall()
            self.tables = [table[0] for table in tables_result]
            conn.close()
            
            # Update table combobox
            self.table_combo['values'] = self.tables
            if self.tables:
                self.table_var.set(self.tables[0])
                self.on_table_selected()
            
            self.status_var.set(f"Loaded database: {os.path.basename(db_path)} ({len(self.tables)} tables)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {str(e)}")
            self.status_var.set("Error loading database")
    
    def on_table_selected(self, event=None):
        """Handle table selection with type detection"""
        if not self.current_db_path or not self.table_var.get():
            return
        
        try:
            conn = duckdb.connect(self.current_db_path)
            table_name = self.table_var.get()
            
            # Get column information with types
            self.column_types = get_column_types(conn, table_name)
            self.columns = list(self.column_types.keys())
            
            # Get only numeric columns for label selection
            numeric_columns = [col for col, info in self.column_types.items() if info['is_numeric']]
            
            # Update label column combobox with only numeric columns
            self.label_combo['values'] = numeric_columns
            if numeric_columns:
                # For stock data, try to find 'close' or use last numeric column
                if 'close' in numeric_columns:
                    self.label_col_var.set('close')
                elif 'target' in numeric_columns:
                    self.label_col_var.set('target')
                else:
                    self.label_col_var.set(numeric_columns[-1])
            
            conn.close()
            
            # Update status with column type info
            numeric_count = len(numeric_columns)
            total_count = len(self.columns)
            self.status_var.set(f"Table selected: {table_name} ({total_count} columns, {numeric_count} numeric)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load table info: {str(e)}")
    
    def preview_data(self):
        """Preview data in the grid"""
        if not self.current_db_path or not self.table_var.get():
            messagebox.showwarning("No Selection", "Please select a database and table first")
            return
        
        try:
            conn = duckdb.connect(self.current_db_path)
            table_name = self.table_var.get()
            
            # Get sample data (limit to 100 rows for preview)
            query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 100"
            result = conn.execute(query).fetchall()
            
            # Clear existing data
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Setup columns
            self.tree['columns'] = self.columns
            self.tree['show'] = 'headings'
            
            # Configure column headings and widths
            for col in self.columns:
                self.tree.heading(col, text=col)
                if col == 'ticker':
                    self.tree.column(col, width=80, minwidth=60)
                elif col == 'timestamp':
                    self.tree.column(col, width=150, minwidth=120)
                elif col in ['open', 'high', 'low', 'close']:
                    self.tree.column(col, width=80, minwidth=60)
                elif col == 'vol':
                    self.tree.column(col, width=100, minwidth=80)
                else:
                    self.tree.column(col, width=100, minwidth=50)
            
            # Insert data with formatting
            for row in result:
                formatted_row = []
                for i, value in enumerate(row):
                    if self.columns[i] == 'timestamp':
                        # Format timestamp
                        formatted_row.append(str(value)[:19] if value else "")
                    elif self.columns[i] in ['open', 'high', 'low', 'close']:
                        # Format prices to 2 decimal places
                        formatted_row.append(f"{float(value):.2f}" if value else "")
                    elif self.columns[i] == 'vol':
                        # Format volume with commas
                        formatted_row.append(f"{int(value):,}" if value else "")
                    else:
                        formatted_row.append(str(value) if value else "")
                self.tree.insert('', 'end', values=formatted_row)
            
            conn.close()
            self.status_var.set(f"Preview: {len(result)} rows displayed (max 100)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview data: {str(e)}")
    
    def load_data(self):
        """Load data with enhanced type handling"""
        if not self.current_db_path or not self.table_var.get() or not self.label_col_var.get():
            messagebox.showwarning("Incomplete Selection", 
                                 "Please select database, table, and label column")
            return
        
        try:
            # Get feature columns (only numeric columns, excluding label and system columns)
            label_col = self.label_col_var.get()
            
            # Get all numeric columns except the label
            numeric_columns = [col for col, info in self.column_types.items() if info['is_numeric']]
            
            # Exclude label column and common system columns
            excluded_cols = [label_col.lower(), 'ticker', 'symbol', 'timestamp', 'date', 'time', 'id', 'format']
            feature_cols = [col for col in numeric_columns 
                          if col.lower() not in excluded_cols]
            
            if not feature_cols:
                messagebox.showerror("Error", "No numeric feature columns available")
                return
            
            print(f"Selected feature columns: {feature_cols}")
            print(f"Selected label column: {label_col}")
            
            # Load data using enhanced function
            X, y = load_data_from_duckdb_enhanced(
                db_path=self.current_db_path,
                table_name=self.table_var.get(),
                feature_columns=feature_cols,
                label_column=label_col
            )
            
            if X is not None and y is not None:
                self.loaded_X = X
                self.loaded_y = y
                self.status_var.set(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
                messagebox.showinfo("Success", 
                                  f"Data loaded successfully!\n"
                                  f"Samples: {X.shape[0]}\n"
                                  f"Features: {X.shape[1]} ({', '.join(feature_cols)})\n"
                                  f"Label: {label_col}\n"
                                  f"Ready for training.")
            else:
                messagebox.showerror("Error", "Failed to load data")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def use_data_and_close(self):
        """Use the loaded data and close GUI"""
        if self.loaded_X is None or self.loaded_y is None:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        self.root.quit()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()
        return self.loaded_X, self.loaded_y

def launch_enhanced_gui_loader():
    """Launch the enhanced GUI data loader"""
    print("Launching Enhanced DuckDB Stock Data GUI loader...")
    gui = EnhancedDuckDBLoaderGUI()
    return gui.run()

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

