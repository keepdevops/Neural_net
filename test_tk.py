# Make sure you're in the correct directory
cd /Users/porupine/neural_net/net

# Create the test file
cat > test_tk.py << 'EOL'
import tkinter as tk
from tkinter import ttk, messagebox
import sys

def main():
    try:
        print("Starting Tkinter test...")
        print(f"Python version: {sys.version}")
        
        root = tk.Tk()
        print("Created root window")
        
        root.title("Minimal Test")
        print("Set window title")
        
        def on_click():
            print("Button clicked")
            try:
                messagebox.showinfo("Test", "Button clicked")
                print("Showed message box")
            except Exception as e:
                print(f"Error showing message box: {e}")
                import traceback
                traceback.print_exc()
        
        button = ttk.Button(root, text="Click Me", command=on_click)
        print("Created button")
        
        button.pack(padx=20, pady=20)
        print("Packed button")
        
        print("Starting main loop...")
        root.mainloop()
        print("Main loop ended")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
EOL
