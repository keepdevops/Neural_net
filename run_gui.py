#!/usr/bin/env python3
import tkinter as tk
from data_visualization.data_visualization_gui import DataVisualizationGUI

def launch_gui():
    """Launch the data visualization GUI application"""
    try:
        # Create the root window
        root = tk.Tk()
        
        # Set window size and position
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create the main application
        app = DataVisualizationGUI(root)
        
        # Start the main event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error launching GUI: {str(e)}")
        raise

if __name__ == "__main__":
    launch_gui()
