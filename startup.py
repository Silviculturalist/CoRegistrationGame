import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize_scalar

# Import the new modular components:
from trees import Stand, SavedStand
from chm_plot import CHMPlot, SavedPlot, PlotCenters, plot_height_curve
# Note: the main application is launched from app.App

def Naslund1936kwargs(diameter, *params):
    return 1.3 + (diameter / (params[0] + params[1] * diameter)) ** params[2]

def getDiameter(height, *params):
    def findDiameter(height, *params):
        def objective(x):
            return (height - Naslund1936kwargs(x, *params)) ** 2
        return minimize_scalar(objective, bounds=(0, 100), method='bounded').x
    return min(findDiameter(height, *params), 1.5)

def update_column_options(file_path, sep, comboboxes, mapping_vars):
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path, sep=sep)
            columns = df.columns.tolist()
            # Reset mapping variables
            for var in mapping_vars:
                mapping_vars[var].set('')
            # Update each combobox with the file columns
            for combobox in comboboxes.values():
                combobox['values'] = columns
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read columns from file: {e}")
    else:
        messagebox.showerror("Invalid File", "Please select a CSV file.")

def plot_params_graph(*args):
    try:
        params = [float(param_entry_1.get()),
                  float(param_entry_2.get()),
                  float(param_entry_3.get())]
        fig = plot_height_curve(params)
        # Clear existing graph
        for widget in graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    except ValueError:
        pass

def start_program(id, file_path, chm_path, file_column_mapping, chm_column_mapping, 
                  file_sep, chm_sep, file_calculate_dbh, file_calculate_h, 
                  chm_calculate_dbh, chm_calculate_h, params, output_folder):
    # Create necessary directories if they don't exist
    if not os.path.isdir('./Transformations'):
        os.mkdir('./Transformations')
    if not os.path.isdir('./Trees'):
        os.mkdir('./Trees')

    # Pass the mapping and separator to the Stand initializer.
    MyData = Stand(ID=id, file_path=file_path, mapping=file_column_mapping, sep=file_sep)
    # Optionally, update CHMPlot similarly if needed:
    MyCHM = CHMPlot(file_path=chm_path, x=MyData.center[0], y=MyData.center[1], dist=70, mapping=chm_column_mapping,sep=chm_sep)

    MyPlotCenters = PlotCenters(MyData)
    
    from app import App
    root = tk.Tk()
    app = App(root, MyData, MyCHM, MyPlotCenters)
    root.mainloop()


def toggle_file_impute_dbh():
    if file_dbh_calc_var.get():
        file_h_calc_var.set(False)
    update_params_section()

def toggle_file_impute_h():
    if file_h_calc_var.get():
        file_dbh_calc_var.set(False)
    update_params_section()

def toggle_chm_impute_dbh():
    if chm_dbh_calc_var.get():
        chm_h_calc_var.set(False)
    update_params_section()

def toggle_chm_impute_h():
    if chm_h_calc_var.get():
        chm_dbh_calc_var.set(False)
    update_params_section()


def on_start():
    id_val = id_entry.get()
    file_path_val = file_path_entry.get()
    chm_path_val = chm_path_entry.get()
    output_folder_val = output_folder_entry.get()
    file_sep_val = file_sep_var.get()
    chm_sep_val = chm_sep_var.get()

    file_path_val = file_path_entry.get().strip()
    chm_path_val = chm_path_entry.get().strip()
    
    # Check if the tree data file exists
    if not os.path.exists(file_path_val):
        messagebox.showerror("File Not Found", f"Tree data file not found:\n{file_path_val}")
        return  # Stop further execution
    
    # Check if the CHM file exists
    if not os.path.exists(chm_path_val):
        messagebox.showerror("File Not Found", f"CHM file not found:\n{chm_path_val}")
        return  # Stop further execution
    
    if not os.path.exists(output_folder_val):
        messagebox.showerror("Folder Not Found", f"Output folder path not found:\n{output_folder_val}")
        return  # Stop further execution



    #Check to avoid overwriting files.
    # Construct the filenames for the trees and transformation files
    trees_filename = f"Stand_{id_val}_trees.csv"
    trans_filename = f"Stand_{id_val}_transformation.csv"
    trees_filepath = os.path.join(output_folder_val, trees_filename)
    trans_filepath = os.path.join('./Transformations', trans_filename)

    # Check if files exist and the checkbox is not checked
    if not allow_overwrite_var.get() and (os.path.exists(trees_filepath) or os.path.exists(trans_filepath)):
        files_at_risk = []
        if os.path.exists(trees_filepath):
            files_at_risk.append(trees_filepath)
        if os.path.exists(trans_filepath):
            files_at_risk.append(trans_filepath)
        warning_message = (
            "Overwrite Warning:\n\n"
            "The following file(s) already exist and will be overwritten:\n"
            + "\n".join(files_at_risk)
            + "\n\nPlease either check the 'Allow overwriting existing files' box or change the Stand ID to a unique value."
        )
        messagebox.showerror("Overwrite Warning", warning_message)
        return  # Do not start the program

    
    file_column_mapping = {
        'StandID': file_mapping_vars['StandID'].get(),
        'PlotID': file_mapping_vars['PlotID'].get(),
        'TreeID': file_mapping_vars['TreeID'].get(),
        'X': file_mapping_vars['X'].get(),
        'Y': file_mapping_vars['Y'].get(),
        'DBH': file_mapping_vars['DBH'].get(),
        'H': file_mapping_vars['H'].get()
    }

    chm_column_mapping = {
        'StandID': chm_mapping_vars['StandID'].get(),
        'PlotID': chm_mapping_vars['PlotID'].get(),
        'TreeID': chm_mapping_vars['TreeID'].get(),
        'X': chm_mapping_vars['X'].get(),
        'Y': chm_mapping_vars['Y'].get(),
        'DBH': chm_mapping_vars['DBH'].get(),
        'H': chm_mapping_vars['H'].get()
    }
    
    file_calculate_dbh = file_dbh_calc_var.get()
    file_calculate_h = file_h_calc_var.get()
    chm_calculate_dbh = chm_dbh_calc_var.get()
    chm_calculate_h = chm_h_calc_var.get()

    params = [float(param_entry_1.get()), 
              float(param_entry_2.get()), 
              float(param_entry_3.get())]
    
    # Pass mapping and separator to Stand
    start_program(id_val, file_path_val, chm_path_val, file_column_mapping, 
                  chm_column_mapping, file_sep_val, chm_sep_val, file_calculate_dbh, 
                  file_calculate_h, chm_calculate_dbh, chm_calculate_h, params, output_folder_val)


def select_file_path():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, file_path)
        update_column_options(file_path, file_sep_var.get(), file_comboboxes, file_mapping_vars)

def select_chm_path():
    chm_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if chm_path:
        chm_path_entry.delete(0, tk.END)
        chm_path_entry.insert(0, chm_path)
        update_column_options(chm_path, chm_sep_var.get(), chm_comboboxes, chm_mapping_vars)

def select_output_folder():
    output_folder = filedialog.askdirectory()
    if output_folder:
        output_folder_entry.delete(0, tk.END)
        output_folder_entry.insert(0, output_folder)

def on_file_sep_change(*args):
    update_column_options(file_path_entry.get(), file_sep_var.get(), file_comboboxes, file_mapping_vars)

def on_chm_sep_change(*args):
    update_column_options(chm_path_entry.get(), chm_sep_var.get(), chm_comboboxes, chm_mapping_vars)

def on_closing():
    root.destroy()
    sys.exit()  # Ensure the program exits completely

# --------------------
# Build the Start-Up Menu UI
# --------------------
root = tk.Tk()
root.title("Co-Registration Game Start-Up Menu")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Stand ID
ttk.Label(root, text="ID:").grid(column=0, row=0, padx=10, pady=5)
id_entry = ttk.Entry(root)
id_entry.grid(column=1, row=0, padx=10, pady=5)
# Create a BooleanVar to track if overwriting is allowed (default False)
allow_overwrite_var = tk.BooleanVar(value=False)

# Add a checkbutton for allowing overwrite in the output folder frame
overwrite_frame = ttk.Frame(root)
overwrite_frame.grid(column=0, row=7, columnspan=2, pady=5, sticky="w")
ttk.Checkbutton(overwrite_frame, text="Allow overwriting existing files", variable=allow_overwrite_var).pack(side=tk.LEFT)


# File Path Frame
file_path_frame = ttk.LabelFrame(root, text="Tree Data File Path")
file_path_frame.grid(column=0, row=1, padx=10, pady=10, sticky="nsew")

ttk.Label(file_path_frame, text="File Path:").grid(column=0, row=0, padx=10, pady=5)
file_path_entry = ttk.Entry(file_path_frame)
file_path_entry.grid(column=1, row=0, padx=10, pady=5)
ttk.Button(file_path_frame, text="Browse", command=select_file_path).grid(column=2, row=0, padx=10, pady=5)

ttk.Label(file_path_frame, text="CSV Separator:").grid(column=0, row=1, padx=10, pady=5)
file_sep_var = tk.StringVar(value=",")
file_sep_combobox = ttk.Combobox(file_path_frame, textvariable=file_sep_var, state="readonly", 
                                 values=[",", ";", "\t", "|"])
file_sep_combobox.grid(column=1, row=1, padx=10, pady=5)
file_sep_combobox.bind('<<ComboboxSelected>>', on_file_sep_change)

# Mapping variables for tree file columns
file_mapping_vars = {label: tk.StringVar() for label in ['StandID', 'PlotID', 'TreeID', 'X', 'Y', 'DBH', 'H']}
file_comboboxes = {}
for i, label in enumerate(['StandID', 'PlotID', 'TreeID', 'X', 'Y', 'DBH', 'H']):
    ttk.Label(file_path_frame, text=label).grid(column=0, row=i+2, padx=5, pady=2)
    file_comboboxes[label] = ttk.Combobox(file_path_frame, textvariable=file_mapping_vars[label], state="readonly")
    file_comboboxes[label].grid(column=1, row=i+2, padx=5, pady=2)
    if label == 'DBH':
        file_dbh_calc_var = tk.BooleanVar()
        ttk.Checkbutton(file_path_frame, text="Impute DBH", variable=file_dbh_calc_var,
                        command=toggle_file_impute_dbh).grid(column=2, row=i+2, padx=5, pady=2)
    elif label == 'H':
        file_h_calc_var = tk.BooleanVar()
        ttk.Checkbutton(file_path_frame, text="Impute Height", variable=file_h_calc_var,
                        command=toggle_file_impute_h).grid(column=2, row=i+2, padx=5, pady=2)


# CHM Path Frame
chm_path_frame = ttk.LabelFrame(root, text="CHM Data File Path")
chm_path_frame.grid(column=1, row=1, padx=10, pady=10, sticky="nsew")

ttk.Label(chm_path_frame, text="CHM Path:").grid(column=0, row=0, padx=10, pady=5)
chm_path_entry = ttk.Entry(chm_path_frame)
chm_path_entry.grid(column=1, row=0, padx=10, pady=5)
ttk.Button(chm_path_frame, text="Browse", command=select_chm_path).grid(column=2, row=0, padx=10, pady=5)

ttk.Label(chm_path_frame, text="CSV Separator:").grid(column=0, row=1, padx=10, pady=5)
chm_sep_var = tk.StringVar(value=",")
chm_sep_combobox = ttk.Combobox(chm_path_frame, textvariable=chm_sep_var, state="readonly", 
                                values=[",", ";", "\t", "|"])
chm_sep_combobox.grid(column=1, row=1, padx=10, pady=5)
chm_sep_combobox.bind('<<ComboboxSelected>>', on_chm_sep_change)

# Mapping variables for CHM file columns
chm_mapping_vars = {label: tk.StringVar() for label in ['StandID', 'PlotID', 'TreeID', 'X', 'Y', 'DBH', 'H']}
chm_comboboxes = {}
for i, label in enumerate(['StandID', 'PlotID', 'TreeID', 'X', 'Y', 'DBH', 'H']):
    ttk.Label(chm_path_frame, text=label).grid(column=0, row=i+2, padx=5, pady=2)
    chm_comboboxes[label] = ttk.Combobox(chm_path_frame, textvariable=chm_mapping_vars[label], state="readonly")
    chm_comboboxes[label].grid(column=1, row=i+2, padx=5, pady=2)
    if label == 'DBH':
        chm_dbh_calc_var = tk.BooleanVar()
        ttk.Checkbutton(chm_path_frame, text="Impute DBH", variable=chm_dbh_calc_var,
                        command=toggle_chm_impute_dbh).grid(column=2, row=i+2, padx=5, pady=2)
    elif label == 'H':
        chm_h_calc_var = tk.BooleanVar()
        ttk.Checkbutton(chm_path_frame, text="Impute Height", variable=chm_h_calc_var,
                        command=toggle_chm_impute_h).grid(column=2, row=i+2, padx=5, pady=2)


# Output Folder Frame
output_folder_frame = ttk.LabelFrame(root, text="Output Folder")
output_folder_frame.grid(column=0, row=3, padx=10, pady=10, columnspan=2, sticky="nsew")
ttk.Label(output_folder_frame, text="Output Folder:").grid(column=0, row=0, padx=10, pady=5)
output_folder_entry = ttk.Entry(output_folder_frame)
output_folder_entry.grid(column=1, row=0, padx=10, pady=5)
ttk.Button(output_folder_frame, text="Browse", command=select_output_folder).grid(column=2, row=0, padx=10, pady=5)

# Näslund Parameters Section (shown if any calculate checkbox is enabled)
params_frame = ttk.LabelFrame(root, text="Näslund Parameters")
params_frame.grid(column=0, row=4, columnspan=2, pady=10)
params_frame.grid_remove()

ttk.Label(params_frame, text="Parameter 1:").grid(column=0, row=0, padx=5, pady=2)
param_entry_1 = ttk.Entry(params_frame)
param_entry_1.grid(column=1, row=0, padx=5, pady=2)
param_entry_1.insert(0, "1.74105089")
param_entry_1.bind("<KeyRelease>", plot_params_graph)

ttk.Label(params_frame, text="Parameter 2:").grid(column=0, row=1, padx=5, pady=2)
param_entry_2 = ttk.Entry(params_frame)
param_entry_2.grid(column=1, row=1, padx=5, pady=2)
param_entry_2.insert(0, "0.35979281")
param_entry_2.bind("<KeyRelease>", plot_params_graph)

ttk.Label(params_frame, text="Parameter 3:").grid(column=0, row=2, padx=5, pady=2)
param_entry_3 = ttk.Entry(params_frame)
param_entry_3.grid(column=1, row=2, padx=5, pady=2)
param_entry_3.insert(0, "3.56879791")
param_entry_3.bind("<KeyRelease>", plot_params_graph)

# Graph Frame to display the Näslund height curve
graph_frame = ttk.Frame(root)
graph_frame.grid(column=0, row=5, columnspan=2, pady=10)
graph_frame.grid_remove()

def update_params_section():
    # Show parameters and graph if any calculation is enabled
    if file_dbh_calc_var.get() or file_h_calc_var.get() or chm_dbh_calc_var.get() or chm_h_calc_var.get():
        params_frame.grid()
        graph_frame.grid()
        plot_params_graph()
    else:
        params_frame.grid_remove()
        graph_frame.grid_remove()

# Bind the calculate checkboxes to update the params section
file_dbh_calc_var.trace_add("write", lambda *args: update_params_section())
file_h_calc_var.trace_add("write", lambda *args: update_params_section())
chm_dbh_calc_var.trace_add("write", lambda *args: update_params_section())
chm_h_calc_var.trace_add("write", lambda *args: update_params_section())

# Start Button
start_button = ttk.Button(root, text="Start", command=on_start)
start_button.grid(column=0, row=6, columnspan=2, pady=10)

root.mainloop()
