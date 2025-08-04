"""
2025-05-07
added option for cmd arguments. Now, script can be run with:
python FF-Propeller Picker.py - runs a GUI to select the input file and folder path
python FF-Propeller Picker.py <Input_FF.xlsx> --yes - uses default folder path for dynamic data
python FF-Propeller Picker.py <Input_FF.xlsx> <dynamic_data_folder_path> - uses the specified folder path for dynamic data folder path
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import simpledialog, messagebox
import tkinter as tk

def get_folder_path(default_folder="Dynamic_data"):
    """
    Function to prompt user for folder path if not provided via command-line arguments.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    user_response = messagebox.askyesno("Folder Path", f"Do you want to keep the default folder path '{default_folder}'?")
    if user_response:
        folder_path = default_folder
    else:
        folder_path = askdirectory(title="Select a Folder")
        if not folder_path:
            messagebox.showwarning("Warning", "No folder selected. Using default path.")
            folder_path = default_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Default folder path
default_folder = "Dynamic_data"

# Check for command-line arguments
if len(sys.argv) > 1:
    file_name = sys.argv[1]  # First argument is the file_name
    if len(sys.argv) > 2:
        folder_path_arg = sys.argv[2]  # Second argument is the folder_path
        if folder_path_arg.lower() == "--yes":
            folder_path = default_folder
        else:
            folder_path = folder_path_arg
    else:
        # If folder_path is not provided, ask for confirmation
        folder_path = get_folder_path(default_folder)
else:
    # If no arguments are provided, use tkinter to get file_name and folder_path
    file_name = askopenfilename(title="Select a Input_FF.xlsx file", 
                                filetypes=[("Input_FF", "*Input_FF.xlsx"), ("Excel sheets", "*.xlsx"), ("All files", "*.*")])
    folder_path = get_folder_path(default_folder)

def linterp2(rX, rY, X):
    """
    Linear interpolator / extrapolator
    rX: list or numpy array of known x values
    rY: list or numpy array of known y values
    X: the x value to interpolate/extrapolate
    """
    nR = len(rX)
    if nR < 2:
        return None

    l1, l2 = None, None
    if X < rX[0]:
        l1, l2 = 0, 1
    elif X > rX[-1]:
        l1, l2 = nR - 2, nR - 1
    else:
        for LR in range(nR):
            if rX[LR] == X:
                return rY[LR]
            elif rX[LR] > X:
                l1, l2 = LR, LR - 1
                break
    if l1 is None or l2 is None:
        return None
    return rY[l1] + (rY[l2] - rY[l1]) * (X - rX[l1]) / (rX[l2] - rX[l1])

# Read input file and display dataframe
df_input = pd.read_excel(file_name)
print("\n", df_input, "\n")

plot_in_one = "Yes"  # <-------------------------------------------------------------------# If "Yes", plot all curves in one plot
plot_3D = "Yes"  # <-------------------------------------------------------------------# If "Yes", plot 3D surface

# Get the dynamic data folder and create a folder for saving plots
dynamic_data = folder_path
plots_folder = str(file_name.split("Input_FF.xlsx")[0]) + "Plots"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Prepare output DataFrames based on "Operating point"
operating_points = df_input["Operating point"]
df_thrust = pd.DataFrame(operating_points, columns=["Operating point"])
df_tip_speed = pd.DataFrame(operating_points, columns=["Operating point"])
df_prop_efficiency = pd.DataFrame(operating_points, columns=["Operating point"])
df_mech_power = pd.DataFrame(operating_points, columns=["Operating point"])
df_rpm = pd.DataFrame(operating_points, columns=["Operating point"])

# These lists will collect values for the final 3D plot later.
bigplot_dia = []
bigplot_pitch = []
bigplot_eff = []
bigplot_p2d = []
bigplot_advr = []
filenames = []
bigplot_labels = []

# If plotting all in one plot, create the figure and axes beforehand.

fig_all, ax1_all = plt.subplots(figsize=(10, 6))

# Only one y-axis (for efficiency)
ax1_all.set_xlabel('Advance Ratio')
ax1_all.set_ylabel('Propeller Efficiency')
ax1_all.set_ylim(0, 1)

    
# Iterate over all Excel files in the dynamic_data folder
for filename in os.listdir(dynamic_data):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(dynamic_data, filename)
        df_prop = pd.read_excel(file_path)
        # sort the DataFrame by RPM (lowest to highest)
        df_prop = df_prop.sort_values(by='Advance ratio')
        print(df_prop)
        filenames.append(filename.split(".")[0])
        # Remove rows where efficiency is out of bounds (not between 0 and 1)
        df_prop = df_prop[(df_prop['Propulsion efficiency'] >= 0) & (df_prop['Propulsion efficiency'] < 1)]
        print(f"DataFrame loaded from {filename}:")
        print(df_prop)

        # Extract propeller parameters and data
        diameter = df_prop['Diameter'].iloc[0]  # Assuming constant per file
        pitch = float(filename.split("x")[1].split(" ")[0].split("-")[0].split("_")[0])
        p2d = pitch/diameter

        advance_ratios = df_prop['Advance ratio'].to_numpy()
        cts = df_prop['Ct'].to_numpy()
        cps = df_prop['Cp'].to_numpy()
        effs = df_prop['Propulsion efficiency'].to_numpy()
        final_advance_ratios = []
        adv_ratio_lim = np.max(advance_ratios)
        print("\n", filename)
        print("\n", cts, advance_ratios)

        # Loop over each operating point from the input file.
        for index, row in df_input.iterrows():
            speed = row['Speed (TAS)[m/s]']
            air_density = row['Air density [kg/m3]']
            required_thrust = row['Required thrust [N]']
            try:
                rpm_sim = df_prop['RPM'].iloc[0]
            except KeyError:
                # IF RPM not found replace with RPM calculated from max advance ratio
                rpm_sim = 60 * speed / (advance_ratios[-1] * diameter * 0.0254)
                print(f"RPM not found in {filename}, using calculated RPM: {rpm_sim}")
            alt = row['Altitude (m)']

            print(f"\nOperating point: {row['Operating point']}")

            def thrust_difference(rpm):
                # Calculate advance ratio from current rpm value.
                advance_ratio = 60 * speed / (rpm * diameter * 0.0254)
                ct = linterp2(advance_ratios, cts, advance_ratio)
                if ct is None:
                    return float('inf')
                thrust = air_density * ct * (rpm / 60)**2 * (diameter * 0.0254)**4
                # DEBUG: Print intermediate values if needed
                print(f"TAS: {speed}, RPM: {rpm}, Advance Ratio: {advance_ratio}, Ct: {ct}, Thrust: {thrust}, Required Thrust: {required_thrust}")
                return abs(thrust - required_thrust)
            
            # Initial guess for RPM
            x0 = np.array([rpm_sim])
            print("Initial guess:", x0)
            constraints = ()  
            bounds = [(0, None)]  

            # Try several starting factors to help convergence
            for factor in np.arange(0.1, 2.0, 0.1):
                result = minimize(thrust_difference, x0 * factor, method='SLSQP', bounds=bounds, constraints=constraints)
                # Compute advance_ratio based on the result
                advance_ratio = 60 * speed / (result.x * diameter * 0.0254)
                prop_eff = linterp2(advance_ratios, effs, advance_ratio) * 100
                print("Thrust difference:", result.fun)
                print("RPM:", result.x)
                # Check if the result is valid and within bounds
                if result.success and result.fun < 1e-3 and 0 < advance_ratio < adv_ratio_lim:
                    break


            if result.success == False:
                df_thrust.at[index, filename] = "Did not converge"
                df_tip_speed.at[index, filename] = "Did not converge"
                df_prop_efficiency.at[index, filename] = "Did not converge"
                df_mech_power.at[index, filename] = "Did not converge"
                df_rpm.at[index, filename] = "Did not converge"
                continue

            final_advance_ratios.append(advance_ratio[0])
            ct = linterp2(advance_ratios, cts, advance_ratio)
            if ct is None:
                df_thrust.at[index, filename] = "Interpolation failed"
                df_tip_speed.at[index, filename] = "Interpolation failed"
                df_prop_efficiency.at[index, filename] = "Interpolation failed"
                df_mech_power.at[index, filename] = "Interpolation failed"
                df_rpm.at[index, filename] = "Interpolation failed"
                continue

            thrust = air_density * ct * (result.x / 60)**2 * (diameter * 0.0254)**4
            prop_eff = linterp2(advance_ratios, effs, advance_ratio)*100
            cp = linterp2(advance_ratios, cps, advance_ratio)
            power = air_density * cp * (result.x / 60)**3 * (diameter * 0.0254)**5
            tip_mach = ((result.x * 2 * np.pi / 60) * (diameter*0.0254/2)) / (np.sqrt(287*1.4*(273.15+15-0.0065*alt)))

            df_thrust.at[index, filename] = round(thrust[0], 2)
            df_tip_speed.at[index, filename] = round(tip_mach[0], 2)
            if np.isscalar(prop_eff):
                df_prop_efficiency.at[index, filename] = round(prop_eff, 2)
            else:
                df_prop_efficiency.at[index, filename] = round(prop_eff[0], 2)
            df_mech_power.at[index, filename] = round(power[0], 2)
            df_rpm.at[index, filename] = round(result.x[0], 1)

        # Plotting section for each propeller file
        # If plotting individually, create a separate figure

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Advance Ratio')
        ax1.set_ylabel('Ct and Cp', color=color)
        ax1.plot(advance_ratios, cts, label='Ct', color='tab:blue')
        ax1.plot(advance_ratios, cps, label='Cp', color='tab:cyan')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Propeller Efficiency', color=color)  
        ax2.plot(advance_ratios, effs, label='Propeller Efficiency', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)
        for adv_ratio in final_advance_ratios:
            ax1.axvline(x=adv_ratio, color='tab:green', linestyle='--')
        fig.tight_layout()  
        plt.title(f'Propeller Data from {filename}')

        # Collect handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Combine them
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2

        # Now place the combined legend
        ax1.legend(all_handles, all_labels, loc="lower left", bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes)

        # Save the individual plot
        plot_path = os.path.join(plots_folder, f"{filename.split('.xlsx')[0]}_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)

        # Get the current color from the plot cycle
        color = ax1_all._get_lines.get_next_color()

        # Plot efficiency with the assigned color
        ax1_all.plot(advance_ratios, effs, label=f'Efficiency {filename}', linestyle='-', alpha=0.8, color=color)

        # Plot dashed lines in the same color
        for adv_ratio in final_advance_ratios:
            # Interpolate efficiency at the given advance ratio
            efficiency_at_adv_ratio = np.interp(adv_ratio, advance_ratios, effs)
            ax1_all.axvline(x=adv_ratio, color=color, linestyle='--', alpha=0.5)
            # Plot intersection point as a marker
            ax1_all.scatter(adv_ratio, efficiency_at_adv_ratio, color=color, marker='o', edgecolors='black', zorder=3)

            # Collect data for 3D plotting
            bigplot_dia.append(diameter)
            bigplot_p2d.append(p2d)
            bigplot_advr.append(adv_ratio)
            bigplot_eff.append(prop_eff)
            bigplot_labels.append(filename.split(".xlsx")[0])

# Close any open figures if needed
plt.close('all')

# Finalize the combined plot
ax1_all.set_title('Combined Efficiency Plot')
ax1_all.grid(True)
ax1_all.set_xlim(0,effs.max() * 1.2)  # Set x-axis limits to 0 to max efficiency + 10%
ax1_all.legend(loc="lower center", bbox_to_anchor=(0.75, 0), bbox_transform=ax1.transAxes)
combined_plot_path = os.path.join(plots_folder, "combined_efficiency_plot.png")
fig_all.tight_layout()
fig_all.savefig(combined_plot_path)
#set x-axis limits

# --- Continue with the rest of your script for 3D plotting and saving output Excel files ---



with pd.ExcelWriter(str(file_name.split("Input_FF.xlsx")[0])+"/Output_FF.xlsx", engine='xlsxwriter') as writer:
    start_row = 0
    worksheet = writer.book.add_worksheet("Sheet1")
    writer.sheets["Sheet1"] = worksheet

    # Define a format for wrapping text, making it bold, and adding borders
    wrap_bold_border_format = writer.book.add_format({'text_wrap': True, 'bold': True, 'border': 1, 'valign': 'top'})
    border_format = writer.book.add_format({'border': 1})

    # Define a format for section titles (no borders, no wrapping)
    section_title_format = writer.book.add_format({'bold': False})

    # Remove ".xlsx" from the column names in all DataFrames
    df_thrust.columns = [col.replace('.xlsx', '') for col in df_thrust.columns]
    df_tip_speed.columns = [col.replace('.xlsx', '') for col in df_tip_speed.columns]
    df_prop_efficiency.columns = [col.replace('.xlsx', '') for col in df_prop_efficiency.columns]
    df_mech_power.columns = [col.replace('.xlsx', '') for col in df_mech_power.columns]
    df_rpm.columns = [col.replace('.xlsx', '') for col in df_rpm.columns]

    # Write the "Thrust Data" section
    worksheet.write(start_row, 0, "Thrust Data [N]", section_title_format)  # Use section_title_format here
    start_row += 1
    df_thrust.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)

    # Apply wrap, bold, and border format to the header row of the "Thrust Data" section
    for col_num, value in enumerate(df_thrust.columns):
        worksheet.write(start_row, col_num, value, wrap_bold_border_format)

    # Apply border format to all data cells in the "Thrust Data" section
    for row_num in range(start_row + 1, start_row + 1 + len(df_thrust)):
        for col_num in range(len(df_thrust.columns)):
            worksheet.write(row_num, col_num, df_thrust.iloc[row_num - start_row - 1, col_num], border_format)

    start_row += len(df_thrust) + 3

    # Write the "Tip Speed Data" section
    worksheet.write(start_row, 0, "Tip Speed Data [M]", section_title_format)  # Use section_title_format here
    start_row += 1
    df_tip_speed.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)

    # # Determine the cruise_tip_speed cell (first data cell in the "Tip Speed Data" section)
    # cruise_tip_speed_cell = f"J{start_row + 1}"  # Assuming column J contains the cruise tip speed for the first propeller

    # # Apply conditional formatting to the entire region (A1 to the last row and last column)
    # last_row = start_row + len(df_rpm)  # Assuming df_rpm is the last DataFrame written
    # last_col = len(df_rpm.columns) + 1  # +1 to account for the "Operating point" column

    # worksheet.conditional_format(
    #     f"A1:{chr(64 + last_col)}{last_row}",  # Adjusts the range dynamically
    #     {
    #         'type': 'formula',
    #         'criteria': f"={cruise_tip_speed_cell}>=0.7",
    #         'format': writer.book.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})  # Example: red background
    #     }
    # )

    # Apply wrap, bold, and border format to the header row of the "Tip Speed Data" section
    for col_num, value in enumerate(df_tip_speed.columns):
        worksheet.write(start_row, col_num, value, wrap_bold_border_format)

    # Apply border format to all data cells in the "Tip Speed Data" section
    for row_num in range(start_row + 1, start_row + 1 + len(df_tip_speed)):
        for col_num in range(len(df_tip_speed.columns)):
            worksheet.write(row_num, col_num, df_tip_speed.iloc[row_num - start_row - 1, col_num], border_format)

    start_row += len(df_tip_speed) + 3

    # Write the "Propeller Efficiency Data" section
    worksheet.write(start_row, 0, "Propeller Efficiency Data [%]", section_title_format)  # Use section_title_format here
    start_row += 1
    df_prop_efficiency.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)

    # Apply wrap, bold, and border format to the header row of the "Propeller Efficiency Data" section
    for col_num, value in enumerate(df_prop_efficiency.columns):
        worksheet.write(start_row, col_num, value, wrap_bold_border_format)

    # Apply border format to all data cells in the "Propeller Efficiency Data" section
    for row_num in range(start_row + 1, start_row + 1 + len(df_prop_efficiency)):
        for col_num in range(len(df_prop_efficiency.columns)):
            worksheet.write(row_num, col_num, df_prop_efficiency.iloc[row_num - start_row - 1, col_num], border_format)

    start_row += len(df_prop_efficiency) + 3

    # Write the "Mechanical Power Data" section
    worksheet.write(start_row, 0, "Mechanical Power Data [W]", section_title_format)  # Use section_title_format here
    start_row += 1
    df_mech_power.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)

    # Apply wrap, bold, and border format to the header row of the "Mechanical Power Data" section
    for col_num, value in enumerate(df_mech_power.columns):
        worksheet.write(start_row, col_num, value, wrap_bold_border_format)

    # Apply border format to all data cells in the "Mechanical Power Data" section
    for row_num in range(start_row + 1, start_row + 1 + len(df_mech_power)):
        for col_num in range(len(df_mech_power.columns)):
            worksheet.write(row_num, col_num, df_mech_power.iloc[row_num - start_row - 1, col_num], border_format)

    start_row += len(df_mech_power) + 3

    # Write the "RPM Data" section
    worksheet.write(start_row, 0, "RPM Data [min-1]", section_title_format)  # Use section_title_format here
    start_row += 1
    df_rpm.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)

    # Apply wrap, bold, and border format to the header row of the "RPM Data" section
    for col_num, value in enumerate(df_rpm.columns):
        worksheet.write(start_row, col_num, value, wrap_bold_border_format)

    # Apply border format to all data cells in the "RPM Data" section
    for row_num in range(start_row + 1, start_row + 1 + len(df_rpm)):
        for col_num in range(len(df_rpm.columns)):
            worksheet.write(row_num, col_num, df_rpm.iloc[row_num - start_row - 1, col_num], border_format)


if plot_3D == "Yes":
    # 3D plot section (remains unchanged)
    bigplot_eff = [arr.item() if hasattr(arr, 'item') else arr for arr in bigplot_eff]
    print(bigplot_eff)
    labels = bigplot_labels
    scatter = go.Scatter3d(
        x=bigplot_dia,
        y=bigplot_p2d,
        z=bigplot_eff,
        mode='markers+text',
        text=labels,
        textposition='top center',
        name='Data Points'
    )
    bigplot_dia = np.array(bigplot_dia)
    bigplot_p2d = np.array(bigplot_p2d)
    bigplot_eff = np.array(bigplot_eff)
    filtered_indices = (bigplot_eff >= 0) & (bigplot_eff <= 100)
    filtered_dia = bigplot_dia[filtered_indices]
    filtered_p2d = bigplot_p2d[filtered_indices]
    filtered_eff = bigplot_eff[filtered_indices]
    print("\nThrust DataFrame:\n", df_thrust)
    print("\nTip Speed DataFrame:\n", df_tip_speed)
    print("\nPropeller Efficiency DataFrame:\n", df_prop_efficiency)
    print("\nMechanical Power DataFrame:\n", df_mech_power)
    print("\nRPM DataFrame:\n", df_rpm)

    grid_x = np.linspace(min(filtered_dia), max(filtered_dia), 100)
    grid_y = np.linspace(min(filtered_p2d), max(filtered_p2d), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((filtered_dia, filtered_p2d), filtered_eff, (grid_x, grid_y), method='linear')
    surface = go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        opacity=0.7,
        name='Interpolated Surface'
    )
    fig = go.Figure(data=[scatter, surface])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Diameter [in]'),
            yaxis=dict(title='Pitch to Diameter ratio [-]'),
            zaxis=dict(title='Propeller Efficiency [%]', range=[50, 100])
        ),
        updatemenus=[{
            "buttons": [
                {"args": ["visible", [True, True]], "label": "Show Surface", "method": "restyle"},
                {"args": ["visible", [True, False]], "label": "Hide Surface", "method": "restyle"}
            ],
            "direction": "down",
            "showactive": True
        }]
    )
    fig.show()
