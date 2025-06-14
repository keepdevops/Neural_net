#!/usr/bin/env python3
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime
from pathlib import Path
import threading
import pickle
import duckdb
from data_loader import DataLoaderTab  # Import the new data loader module
from neural_network.trainer import NeuralNetworkTrainer
from neural_network.inference import NeuralNetworkInference

class DataVisualizationGUI:
    """
    Universal GUI for data visualization and neural network training
    Supports DuckDB, CSV, JSON, Parquet, Excel, SQLite, and more
    """
    
    def __init__(self, root):
        """Initialize the Data Visualization GUI"""
        self.root = root
        self.root.title("Neural Network Data Visualization")
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Track unsaved changes
        self.has_unsaved_changes = False
        
        # Set models directory
        self.models_dir = os.path.join("data_files", "models")
        
        # Ensure the root window is properly initialized
        self.root.update_idletasks()
        
        # Initialize variables
        self.current_data = None
        self.training_active = False
        self.recent_files = []  # Initialize recent files list
        
        # Initialize Tkinter variables
        self.file_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        
        # Training variables
        self.lr_var = tk.StringVar(value="0.01")
        self.epochs_var = tk.StringVar(value="1000")
        self.hidden_var = tk.StringVar(value="64")
        self.patience_var = tk.StringVar(value="50")
        self.epoch_var = tk.StringVar(value="Epoch: 0/0")
        self.loss_var = tk.StringVar(value="Loss: N/A")
        self.accuracy_var = tk.StringVar(value="Accuracy: N/A")
        self.progress_var = tk.DoubleVar(value=0)
        
        # Inference variables
        self.selected_model_var = tk.StringVar()
        self.prediction_var = tk.StringVar(value="Prediction: N/A")
        self.probability_var = tk.StringVar(value="Probability: N/A")
        
        # Create matplotlib figures
        self.fig = Figure(figsize=(8, 6))
        self.results_fig = Figure(figsize=(10, 4))
        
        # Initialize trainer and inference engine
        self.trainer = NeuralNetworkTrainer(gui_callback=self.handle_training_callback)
        self.inference = NeuralNetworkInference(gui_callback=self.handle_inference_callback)
        
        # Setup GUI
        self.setup_styling()
        self.setup_gui()
        
        # Create menu bar
        self.create_menu()

    def setup_styling(self):
        """Setup GUI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        
    def setup_gui(self):
        """Setup the main GUI components"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tab frames first
        self.training_tab = ttk.Frame(self.notebook)
        self.inference_tab = ttk.Frame(self.notebook)
        
        # Add tab frames to notebook with icons
        self.notebook.add(self.training_tab, text="🧠 Training")
        self.notebook.add(self.inference_tab, text="🔮 Inference")
        
        # Create data loader (will be first tab)
        self.data_loader = DataLoaderTab(self.root, self.notebook, self.on_data_loaded)
        
        # Create training tab contents
        self.create_training_tab()
        
        # Create inference tab contents
        self.create_inference_tab()
        
        # Create status bar
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_training_tab(self):
        """Create neural network training tab"""
        # Training configuration
        config_frame = ttk.LabelFrame(self.training_tab, text="Training Configuration", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Learning rate section
        lr_frame = ttk.LabelFrame(config_frame, text="Learning Rate Settings", padding="5")
        lr_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lr_frame, text="Initial Learning Rate:").grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5)
        
        # Training parameters
        params_frame = ttk.Frame(config_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Hidden Size:").grid(row=0, column=2, padx=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.hidden_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(params_frame, text="Patience:").grid(row=0, column=4, padx=5, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.patience_var, width=10).grid(row=0, column=5, padx=5)
        
        # Training controls
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.train_button = ttk.Button(control_frame, text="🚀 Start Training", 
                                     command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Training", 
                                    command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Model management section
        model_frame = ttk.LabelFrame(self.training_tab, text="Model Management", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model name entry
        name_frame = ttk.Frame(model_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Model Name:").pack(side=tk.LEFT, padx=5)
        self.model_name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.model_name_var, width=20)
        name_entry.pack(side=tk.LEFT, padx=5)
        
        # Add placeholder text
        name_entry.insert(0, "Enter model name")
        name_entry.bind('<FocusIn>', lambda e: name_entry.delete(0, tk.END) if name_entry.get() == "Enter model name" else None)
        name_entry.bind('<FocusOut>', lambda e: name_entry.insert(0, "Enter model name") if not name_entry.get() else None)
        
        # Model buttons
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.save_button = ttk.Button(button_frame, text="💾 Save Model", 
                                    command=self.save_model, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = ttk.Button(button_frame, text="📂 Load Model", 
                                    command=self.load_model)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Add refresh button
        self.refresh_button = ttk.Button(button_frame, text="🔄 Refresh Models",
                                       command=self.refresh_model_list)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Model list
        list_frame = ttk.LabelFrame(model_frame, text="Saved Models", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for model list
        columns = ('name', 'input_size', 'hidden_size', 'final_loss', 'final_accuracy', 'epochs')
        self.model_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=5)
        
        # Define headings
        self.model_tree.heading('name', text='Model Name')
        self.model_tree.heading('input_size', text='Input Size')
        self.model_tree.heading('hidden_size', text='Hidden Size')
        self.model_tree.heading('final_loss', text='Final Loss')
        self.model_tree.heading('final_accuracy', text='Accuracy')
        self.model_tree.heading('epochs', text='Epochs')
        
        # Define columns
        self.model_tree.column('name', width=150)
        self.model_tree.column('input_size', width=80)
        self.model_tree.column('hidden_size', width=80)
        self.model_tree.column('final_loss', width=100)
        self.model_tree.column('final_accuracy', width=80)
        self.model_tree.column('epochs', width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to load model
        self.model_tree.bind('<Double-1>', self.on_model_double_click)
        
        # Training progress
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress", padding="10")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Training info
        info_frame = ttk.Frame(progress_frame)
        info_frame.pack(fill=tk.X)
        
        self.epoch_var = tk.StringVar(value="Epoch: 0/0")
        ttk.Label(info_frame, textvariable=self.epoch_var).pack(side=tk.LEFT, padx=10)
        
        self.loss_var = tk.StringVar(value="Loss: N/A")
        ttk.Label(info_frame, textvariable=self.loss_var).pack(side=tk.LEFT, padx=10)
        
        self.accuracy_var = tk.StringVar(value="Accuracy: N/A")
        ttk.Label(info_frame, textvariable=self.accuracy_var).pack(side=tk.LEFT, padx=10)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_tab, text="Training Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.training_log = tk.Text(log_frame, height=15, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scroll.set)
        
        self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh model list
        self.refresh_model_list()

    def create_inference_tab(self):
        """Create neural network inference tab"""
        # Model selection
        model_frame = ttk.LabelFrame(self.inference_tab, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model dropdown and refresh
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_select_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(model_select_frame, textvariable=self.selected_model_var, 
                                      state='readonly', width=30)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        # Add refresh button
        self.refresh_inference_button = ttk.Button(model_select_frame, text="🔄 Refresh Models",
                                                 command=self.refresh_inference_models)
        self.refresh_inference_button.pack(side=tk.LEFT, padx=5)
        
        # Bind model selection change
        self.model_combo.bind('<<ComboboxSelected>>', self.on_inference_model_selected)
        
        # Model info
        info_frame = ttk.LabelFrame(model_frame, text="Model Information", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.model_info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        self.model_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Prediction section
        pred_frame = ttk.LabelFrame(self.inference_tab, text="Prediction", padding="10")
        pred_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Prediction controls
        control_frame = ttk.Frame(pred_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.predict_button = ttk.Button(control_frame, text="🔮 Make Prediction", 
                                       command=self.make_prediction, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Prediction results
        results_frame = ttk.Frame(pred_frame)
        results_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(results_frame, textvariable=self.prediction_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(results_frame, textvariable=self.probability_var).pack(side=tk.LEFT, padx=10)
        
        # Prediction log
        log_frame = ttk.LabelFrame(self.inference_tab, text="Inference Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.inference_log = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.inference_log.yview)
        self.inference_log.configure(yscrollcommand=log_scroll.set)
        
        self.inference_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh model list on startup
        self.refresh_inference_models()

    def on_inference_model_selected(self, event=None):
        """Handle when a model is selected in the inference tab"""
        model_name = self.selected_model_var.get()
        if not model_name:
            return
            
        if self.inference.load_model(model_name, save_dir=self.models_dir):
            # Update model info
            model_info = self.inference.get_model_info()
            if model_info:
                info_text = f"Model: {model_info['name']}\n"
                info_text += f"Input Size: {model_info['input_size']}\n"
                info_text += f"Hidden Size: {model_info['hidden_size']}\n"
                info_text += f"Output Size: {model_info['output_size']}"
                
                self.model_info_text.delete('1.0', tk.END)
                self.model_info_text.insert('1.0', info_text)
                
                # Enable prediction button if we have data
                self.predict_button.config(state=tk.NORMAL if self.current_data is not None else tk.DISABLED)
                
                # Log the model load
                self.log_inference_message(f"Model '{model_name}' loaded successfully")

    def refresh_inference_models(self):
        """Refresh the list of available models for inference"""
        try:
            models = self.inference.get_saved_models(save_dir=self.models_dir)
            model_names = [model['name'] for model in models]
            
            # Store current selection
            current_selection = self.selected_model_var.get()
            
            # Update combobox
            self.model_combo['values'] = model_names
            
            # Try to restore previous selection
            if current_selection in model_names:
                self.model_combo.set(current_selection)
            elif model_names:
                self.model_combo.set(model_names[0])
                # Trigger model load for the first model
                self.on_inference_model_selected()
            
            # Log refresh
            self.log_inference_message(f"Model list refreshed: {len(model_names)} models available")
            
        except Exception as e:
            self.log_inference_message(f"Error refreshing models: {str(e)}")

    def log_inference_message(self, message):
        """Add message to inference log"""
        self.inference_log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.inference_log.see(tk.END)

    def make_prediction(self):
        """Make a prediction using the loaded model"""
        if not self.current_data is not None:
            messagebox.showwarning("No Data", "Please load data first")
            return
            
        if not self.inference.model_params:
            messagebox.showwarning("No Model", "Please select a model first")
            return
            
        try:
            # Get prediction
            predictions, probabilities = self.inference.predict(self.current_data)
            
            if predictions is None or probabilities is None:
                raise ValueError("Model returned no predictions")
                
            # Get the last prediction (most recent data point)
            last_pred = float(predictions[-1])  # Convert numpy value to Python float
            last_prob = float(probabilities[-1][0])  # Convert numpy value to Python float
            
            # Update display
            self.prediction_var.set(f"Prediction: {'UP' if last_pred > 0.5 else 'DOWN'}")
            self.probability_var.set(f"Probability: {last_prob:.2%}")
            
            # Log prediction
            self.log_inference_message(f"Made prediction for {len(predictions)} data points")
            self.log_inference_message(f"Latest prediction: {'UP' if last_pred > 0.5 else 'DOWN'} ({last_prob:.2%})")
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.log_inference_message(error_msg)
            messagebox.showerror("Prediction Error", error_msg)

    def handle_training_callback(self, event_type, data):
        """Handle callbacks from the trainer"""
        if event_type == 'progress':
            self.update_training_progress(
                data['epoch'],
                data['loss'],
                data['accuracy'],
                data['progress']
            )
        elif event_type == 'complete':
            self.training_complete(data['final_loss'], data['best_loss'])
        elif event_type == 'error':
            self.training_error(data)
        elif event_type == 'log':
            self.log_message(data)
        elif event_type == 'status':
            self.status_var.set(data)

    def start_training(self):
        """Start neural network training"""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        try:
            # Get training parameters
            learning_rate = float(self.lr_var.get())
            epochs = int(self.epochs_var.get())
            hidden_size = int(self.hidden_var.get())
            patience = int(self.patience_var.get())
            
            # Start training
            self.train_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.trainer.start_training(
                self.current_data,
                learning_rate,
                epochs,
                hidden_size,
                patience
            )
            
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")

    def stop_training(self):
        """Stop neural network training"""
        self.trainer.stop_training()
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_training_progress(self, epoch, loss, accuracy, progress):
        """Update training progress in GUI"""
        self.epoch_var.set(f"Epoch: {epoch}/{self.epochs_var.get()}")
        self.loss_var.set(f"Loss: {loss:.6f}")
        self.accuracy_var.set(f"Accuracy: {accuracy:.2f}%")
        self.progress_var.set(progress)

    def training_complete(self, final_loss, best_loss):
        """Handle training completion"""
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)  # Enable save button after training
        
        # Generate default model name based on timestamp
        default_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_name_var.set(default_name)
        
        self.log_message("Training completed successfully!")
        self.log_message(f"Final Loss: {final_loss:.6f}")
        self.log_message(f"Best Loss: {best_loss:.6f}")
        
        # Update model parameters
        self.model_params = self.trainer.get_model_params()
        self.normalization_params = self.trainer.get_normalization_params()
        
        # Refresh model list
        self.refresh_model_list()

    def training_error(self, error_msg):
        """Handle training error"""
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.log_message(f"Training error: {error_msg}")
        messagebox.showerror("Training Error", f"Training failed: {error_msg}")

    def log_message(self, message):
        """Add message to training log"""
        self.training_log.insert(tk.END, f"{message}\n")
        self.training_log.see(tk.END)

    def save_model(self):
        """Save the current model automatically using the model name from the text field
        
        Returns:
            bool: True if save was successful or not needed, False if cancelled
        """
        if not self.trainer.model_params:
            messagebox.showwarning("No Model", "No model to save")
            return True
            
        try:
            # Get model name from text field
            model_name = self.model_name_var.get().strip()
            if not model_name:
                messagebox.showwarning("No Name", "Please enter a model name")
                return False
                
            # Create filename from model name
            filename = f"{model_name}.pkl"
            filepath = os.path.join(self.models_dir, filename)
            
            # Check if model already exists
            if os.path.exists(filepath):
                if not messagebox.askyesno("Model Exists", 
                    f"Model '{model_name}' already exists. Do you want to overwrite it?"):
                    return False
            
            # Save the model
            if self.trainer.save_model(filepath, save_dir=self.models_dir):
                self.mark_unsaved_changes(False)
                messagebox.showinfo("Success", f"Model '{model_name}' saved successfully")
                # Refresh the model list to show the newly saved model
                self.refresh_model_list()
                return True
            return False
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
            return False

    def load_model(self):
        """Load a saved model"""
        selected = self.model_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a model to load")
            return
            
        model_name = self.model_tree.item(selected[0])['values'][0]
        if self.trainer.load_model(model_name):
            self.model_name_var.set(model_name)
            self.save_button.config(state=tk.NORMAL)
            self.log_message(f"Model '{model_name}' loaded successfully")

    def on_model_double_click(self, event):
        """Handle double-click on model in list"""
        self.load_model()

    def on_data_loaded(self, data, features=None, label=None):
        """Handle when data is loaded or selection changes.
        
        Args:
            data: Loaded pandas DataFrame
            features: List of selected feature columns (optional)
            label: Selected label column (optional)
        """
        self.current_data = data
        if features and label:
            self.feature_columns = features
            self.label_column = label
            self.training_data = data[features + [label]]
        else:
            self.training_data = data
        
        # Update status
        self.status_var.set(f"Data loaded: {len(data)} rows")
        
        # Log data load
        self.log_inference_message(f"Data loaded: {len(data)} rows")
        
        # Enable prediction button if we have a model loaded
        if self.inference.model_params is not None:
            self.predict_button.config(state=tk.NORMAL)
            self.log_inference_message("Model is ready for prediction")

    def handle_inference_callback(self, event_type, data):
        """Handle callbacks from the inference engine"""
        if event_type == 'log':
            self.log_inference_message(data)
        elif event_type == 'error':
            self.log_inference_message(f"Error: {data}")
            messagebox.showerror("Inference Error", str(data))
        elif event_type == 'status':
            self.status_var.set(data)

    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data...", command=self.data_loader.load_data)
        file_menu.add_command(label="Save Model...", command=self.save_model)
        file_menu.add_command(label="Load Model...", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences...", command=self.show_preferences)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)

    def show_preferences(self):
        """Show preferences dialog"""
        # TODO: Implement preferences dialog
        messagebox.showinfo("Preferences", "Preferences dialog coming soon!")

    def show_about(self):
        """Show about dialog"""
        about_text = """
Neural Network Data Visualization
Version 1.0

A tool for training and visualizing neural networks
for time series prediction.

Features:
- Data loading from multiple sources
- Neural network training
- Model inference
- Data visualization
- Model management
"""
        messagebox.showinfo("About", about_text)

    def show_documentation(self):
        """Show documentation"""
        # TODO: Implement documentation viewer
        messagebox.showinfo("Documentation", "Documentation viewer coming soon!")

    def on_closing(self):
        """Handle application exit"""
        if self.training_active:
            if not messagebox.askyesno("Training in Progress", 
                                     "Training is still in progress. Are you sure you want to exit?"):
                return
            self.stop_training()
        
        if self.has_unsaved_changes:
            if not messagebox.askyesno("Unsaved Changes", 
                                     "You have unsaved changes. Do you want to save before exiting?"):
                if not self.save_model():
                    return  # Don't exit if save was cancelled
        else:
            if not messagebox.askyesno("Confirm Exit", "Are you sure you want to exit?"):
                return
        
        # Clean up resources
        try:
            # Close any open files
            if hasattr(self, 'data_loader'):
                self.data_loader.cleanup()
            
            # Close matplotlib figures
            plt.close('all')
            
            # Destroy the root window
            self.root.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during exit: {str(e)}")
            # Force exit if cleanup fails
            self.root.quit()

    def mark_unsaved_changes(self, has_changes=True):
        """Mark that there are unsaved changes
        
        Args:
            has_changes: Whether there are unsaved changes
        """
        self.has_unsaved_changes = has_changes
        # Update window title to show unsaved status
        title = "Neural Network Data Visualization"
        if has_changes:
            title += " *"
        self.root.title(title)

    def refresh_model_list(self):
        """Refresh the list of saved models in the training tab"""
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
            
        # Get list of models
        models = self.trainer.get_saved_models(save_dir=self.models_dir)
        
        # Add models to treeview
        for model in models:
            self.model_tree.insert('', tk.END, values=(
                model['name'],
                model['input_size'],
                model['hidden_size'],
                f"{model['final_loss']:.6f}" if model['final_loss'] is not None else "N/A",
                f"{model['final_accuracy']:.2f}%" if model['final_accuracy'] is not None else "N/A",
                model['epochs_trained']
            ))
            
        # Log refresh
        self.log_message(f"Model list refreshed: {len(models)} models available")

# This file should be imported, not run directly
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Please run 'python run_gui.py' instead.")
