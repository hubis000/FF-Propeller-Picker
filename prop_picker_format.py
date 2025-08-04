import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import re

"Transforms data in pyprop format into format for FF_propeller_picker"
"Diameter can either be in the filename or manually set on line 45"


def get_diameter_from_filename(filepath):
    match = re.search(r'(\d+(?:\.\d+)?)x', filepath)
    if match:
        return float(match.group(1))
    return None

def transform_file(input_file):
    # Determine file type and read into a pandas DataFrame
    if input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        print(f"Unsupported file format: {input_file}")
        return

    # Create a new DataFrame for transformed data
    transformed_df = pd.DataFrame()

    # Copy required columns and calculate new ones
    transformed_df['RPM'] = df.filter(regex='RPM|rpm').iloc[:,0].values
    transformed_df['v_inf'] = df.filter(regex='v_inf').iloc[:,0].values
    transformed_df['Ct'] = df.filter(regex='CtND').iloc[:,0].values
    transformed_df['Cp'] = df.filter(regex='CpND').iloc[:,0].values
    transformed_df['Propulsion efficiency'] = df.filter(regex='Propulsion efficiency').iloc[:,0].values

    # Assuming Diameter is known, let's use 1 for this example
    Diameter = get_diameter_from_filename(input_file)# or 27   # [inch] You can modify this if you have actual diameter values
    # print(f"For file {input_file} using diameter: {Diameter} inch")
    if Diameter is None:
        raise ValueError(f"Could not extract diameter from filename: {input_file}")
    transformed_df['Advance ratio'] = transformed_df['v_inf'] / ((transformed_df['RPM']/60) * (Diameter*0.0254))

    # Add the diameter column (if needed)
    transformed_df['Diameter'] = Diameter

    # Generate the output filename by appending 'trans' to the original file name and setting extension to '.xlsx'
    base_name, _ = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(os.path.dirname(input_file), f"{base_name}_trans.xlsx")

    # Save the transformed DataFrame to the output file as Excel
    transformed_df.to_excel(output_file, index=False)

    print(f"Transformed file saved to: {output_file}")

def transform_excel():
    # Set up the Tkinter root window (it won't show up)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask user to select multiple input files
    input_files = filedialog.askopenfilenames(
        title="Select the input files",
        filetypes=[("Excel and CSV files", "*.xlsx;*.xls;*.csv")]
    )
    if not input_files:
        print("No files selected. Exiting.")
        return

    # Loop through each selected file
    for input_file in input_files:
        transform_file(input_file)

# Run the function
transform_excel()
