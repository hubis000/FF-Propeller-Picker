import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata

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

    if X < rX[0]:  # x < xmin, extrapolate
        l1, l2 = 0, 1

    elif X > rX[-1]:  # x > xmax, extrapolate
        l1, l2 = nR - 2, nR - 1

    else:
        # a binary search would be better here
        for LR in range(nR):
            if rX[LR] == X:  # x is exact from table
                return rY[LR]

            elif rX[LR] > X:  # x is between tabulated values, interpolate
                l1, l2 = LR, LR - 1
                break

    if l1 is None or l2 is None:
        return None

    return rY[l1] + (rY[l2] - rY[l1]) * (X - rX[l1]) / (rX[l2] - rX[l1])


df_input = pd.read_excel("Input.xlsx")
print("\n", df_input, "\n")

# Extract the "Operating point" column
operating_points = df_input["Operating point"]

# Create 4 DataFrames with one column each and rows matching the "Operating point" column
df_thrust = pd.DataFrame(operating_points, columns=["Operating point"])
df_tip_speed = pd.DataFrame(operating_points, columns=["Operating point"])
df_prop_efficiency = pd.DataFrame(operating_points, columns=["Operating point"])
df_mech_power = pd.DataFrame(operating_points, columns=["Operating point"])
df_rpm = pd.DataFrame(operating_points, columns=["Operating point"])

bigplot_dia = []
bigplot_pitch = []
bigplot_eff = []
bigplot_p2d = []

# Define the folder path
dynamic_data = "Dynamic_data"
plots_folder = "Plots"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
# Iterate over all Excel files in the folder
for filename in os.listdir(dynamic_data):
    if filename.endswith(".xlsx"):
        # Load the Excel file into a DataFrame
        file_path = os.path.join(dynamic_data, filename)
        df_prop = pd.read_excel(file_path)
        
        # Print the DataFrame to verify the content
        print(f"DataFrame loaded from {filename}:")
        print(df_prop)

        # Extract necessary columns from df_prop
        diameter = df_prop['Diameter'].iloc[0]  # Assuming diameter is constant
        pitch = float(filename.split("x")[1].split(" ")[0])
        p2d = pitch/diameter
        
        # Convert columns to NumPy arrays
        advance_ratios = df_prop['Advance ratio'].to_numpy()
        cts = df_prop['Ct'].to_numpy()
        cps = df_prop['Cp'].to_numpy()
        effs = df_prop['Propulsion efficiency'].to_numpy()
        final_advance_ratios = []
        print("\n",cts,advance_ratios)

        for index, row in df_input.iterrows():
            speed = row['Speed (TAS)[m/s]']
            air_density = row['Air density [kg/m3]']
            required_thrust = row['Required thrust [N]']
            rpm_sim = df_prop['RPM'].iloc[0]
            alt = row['Altitude (m)']

            def thrust_difference(rpm):
                advance_ratio = 60 * speed / (rpm * diameter * 0.0254)
                ct = linterp2(advance_ratios, cts, advance_ratio)
                if ct is None:
                    return float('inf')  # Return a large value to indicate failure
                thrust = air_density * ct * (rpm / 60)**2 * (diameter * 0.0254)**4

                # DEBUGGING ---------------------
                print(f"TAS: {speed}, RPM: {rpm}, Advance Ratio: {advance_ratio}, Ct: {ct}, Thrust: {thrust}, Required Thrust: {required_thrust}")
                return abs(thrust - required_thrust)
            
            # Initial guess for the variables
            x0 = np.array([rpm_sim])  # Initial guess for RPM

            # Define any constraints (if applicable)
            constraints = ()  # Add constraints here if needed

            # Define any bounds on the variables (if applicable)
            bounds = [(0, None)]  # RPM should be non-negative

            # Perform the optimization
            result = minimize(thrust_difference, x0, method='SLSQP', bounds=bounds, constraints=constraints)

            # Print the results
            print("Thrust difference:", result.fun)
            print("RPM:", result.x)

            advance_ratio = 60 * speed / (result.x * diameter * 0.0254)
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

            # Append results to DataFrames
            df_thrust.at[index, filename] = round(thrust[0],2)
            df_tip_speed.at[index, filename] = round(tip_mach[0],2)
            if np.isscalar(prop_eff):
                df_prop_efficiency.at[index, filename] = round(prop_eff, 2)
            else:
                df_prop_efficiency.at[index, filename] = round(prop_eff[0], 2)
            df_mech_power.at[index, filename] = round(power[0],2)
            df_rpm.at[index, filename] = round(result.x[0], 1)

        # Plotting the data for each propeller
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
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        # Save the plot to the Plots folder
        plot_path = os.path.join(plots_folder, f"{filename.split('.xlsx')[0]}_plot.png")
        plt.savefig(plot_path)

        bigplot_dia.append(diameter)
        bigplot_p2d.append(p2d)
        bigplot_eff.append(prop_eff)
        
plt.close('all')


#3D plot
# bigplot_pitch = [float(p) for p in bigplot_pitch]
bigplot_eff = [arr.item() for arr in bigplot_eff]

print(bigplot_eff)

scatter = go.Scatter3d(
    x=bigplot_dia,
    y=bigplot_p2d,
    z=bigplot_eff,
    mode='markers',
    name='Data Points'
)


grid_x = np.linspace(min(bigplot_dia), max(bigplot_dia), 100)
grid_y = np.linspace(min(bigplot_p2d), max(bigplot_p2d), 100)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)
grid_z = griddata((bigplot_dia, bigplot_p2d), bigplot_eff, (grid_x, grid_y), method='linear')
surface = go.Surface(
    x=grid_x,
    y=grid_y,
    z=grid_z,
    colorscale='Viridis',
    opacity=0.7,
    name='Interpolated Surface'
)

fig = go.Figure(data=[scatter, surface])   #fig = go.Figure(data=[scatter, surface])
fig.update_layout(
    scene=dict(
        xaxis_title='Diameter',
        yaxis_title='Pitch to Diamter ratio',
        zaxis_title='Propeller Efficiency'
    ),
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["visible", [True, True]],
                    label="Show Surface",
                    method="restyle"
                ),
                dict(
                    args=["visible", [True, False]],
                    label="Hide Surface",
                    method="restyle"
                )
            ]),
            direction="down",
            showactive=True
        )
    ]
)

fig.show()


# Print the final DataFrames to verify the results
print("\nThrust DataFrame:\n", df_thrust)
print("\nTip Speed DataFrame:\n", df_tip_speed)
print("\nPropeller Efficiency DataFrame:\n", df_prop_efficiency)
print("\nMechanical Power DataFrame:\n", df_mech_power)
print("\nRPM DataFrame:\n", df_rpm)

# Save the final DataFrames to Output.xlsx with two empty rows separating them and a header
with pd.ExcelWriter("Output.xlsx", engine='xlsxwriter') as writer:
    start_row = 0
    
    worksheet = writer.book.add_worksheet("Sheet1")
    writer.sheets["Sheet1"] = worksheet

    start_row = 0
    
    writer.sheets["Sheet1"].write(start_row, 0, "Thrust Data")
    start_row += 1
    df_thrust.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)
    start_row += len(df_thrust) + 3
    
    writer.sheets["Sheet1"].write(start_row, 0, "Tip Speed Data")
    start_row += 1
    df_tip_speed.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)
    start_row += len(df_tip_speed) + 3

    writer.sheets["Sheet1"].write(start_row, 0, "Propeller Efficiency Data")
    start_row += 1    
    df_prop_efficiency.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)
    start_row += len(df_prop_efficiency) + 3

    writer.sheets["Sheet1"].write(start_row, 0, "Mechanical Power Data")
    start_row += 1    
    df_mech_power.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)
    start_row += len(df_prop_efficiency) + 3    

    writer.sheets["Sheet1"].write(start_row, 0, "RPM Data")
    start_row += 1
    df_rpm.to_excel(writer, sheet_name="Sheet1", startrow=start_row, index=False)
