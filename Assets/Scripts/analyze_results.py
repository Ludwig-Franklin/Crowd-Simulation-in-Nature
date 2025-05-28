import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Fix for colormap
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MultipleLocator

def convert_float(value):
    """Convert comma-formatted number to float"""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def load_data(filename):
    if not os.path.exists(filename):
        return None
        
    # First read the configuration section
    config_lines = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i < 20:  # First 20 lines are config
                config_lines.append(line.strip())
                
    # Then read the data section
    try:
        data = pd.read_csv(filename, delimiter=':', skiprows=25, names=['Step', 'TimeForInterval', 'Efficiency', 'Civility'])
        
        # Convert comma decimals to periods
        for col in ['TimeForInterval', 'Efficiency', 'Civility']:
            data[col] = data[col].apply(lambda x: convert_float(x) if pd.notnull(x) else x)
            
        return data, config_lines
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def adjust_plot_height(ax, fig):
    """Adjust plot height to match legend height if legend is larger than plot"""
    # Draw canvas to compute legend size
    fig.canvas.draw()
    
    # Get legend and plot sizes
    legend = ax.get_legend()
    if legend is None:  # No legend to adjust for
        return
        
    legend_bbox = legend.get_window_extent()
    plot_bbox = ax.get_window_extent()
    
    # Convert to figure coordinates
    legend_height = np.ceil(legend_bbox.transformed(fig.dpi_scale_trans.inverted()).height)
    current_height = fig.get_size_inches()[1]
    current_width = fig.get_size_inches()[0]
    
    # Only increase height if legend is taller than current height
    if legend_height > current_height:
        fig.set_size_inches(current_width, legend_height + 0.2)
    
    plt.tight_layout()

def plot_test1(data_folder):
    """Plot trail formation parameters test results"""
    test_folder = os.path.join(data_folder, "I_and_T_values")
    results_folder = os.path.join(test_folder, "Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # List all files in the data folder
    files = os.listdir(test_folder)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Store T and I values for each simulation
    T_values = set()
    I_values = set()
    results = []  # Store (T, I, efficiency, civility) tuples
    
    # Look for simulation folders
    for sim_folder in files:
        if sim_folder.startswith("Helbing_sim_"):
            result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if result is not None:
                data, config_lines = result
                
                # Extract T and I values from config
                T = None
                I = None
                for line in config_lines:
                    if "Trail Recovery Rate" in line:
                        try:
                            T = convert_float(line.split("# Trail Recovery Rate:")[1].strip())
                            T_values.add(T)
                        except Exception as e:
                            print(f"Error parsing T from line: {line}, Error: {e}")
                    elif "Footstep Intensity" in line:
                        try:
                            I = convert_float(line.split("# Footstep Intensity:")[1].strip())
                            I_values.add(I)
                        except Exception as e:
                            print(f"Error parsing I from line: {line}, Error: {e}")
                
                if T is not None and I is not None:
                    try:
                        # Get final values (skip NaN rows)
                        final_data = data.dropna()
                        if not final_data.empty:
                            final_efficiency = final_data['Efficiency'].iloc[-1]
                            final_civility = final_data['Civility'].iloc[-1] * 20  # Scale civility
                            results.append((T, I, final_efficiency, final_civility))
                    except Exception as e:
                        print(f"Error processing data for T={T}, I={I}: {e}")

    # Convert sets to sorted lists
    T_values = sorted(list(T_values))
    I_values = sorted(list(I_values))
    
    if not T_values or not I_values or not results:
        print("Error: No valid data found for plotting")
        return

    # Create matrices for heatmaps
    efficiency_matrix = np.zeros((len(I_values), len(T_values)))
    civility_matrix = np.zeros((len(I_values), len(T_values)))
    
    # Fill matrices
    for T, I, eff, civ in results:
        try:
            i = I_values.index(I)
            j = T_values.index(T)
            efficiency_matrix[i,j] = convert_float(eff)
            civility_matrix[i,j] = convert_float(civ)
        except Exception as e:
            print(f"Error processing result T={T}, I={I}: {e}")

    # For efficiency time series
    fig = plt.figure(figsize=(12, 3))  # Start with a wider, shorter figure
    ax = plt.gca()
    
    for sim_folder in files:
        if sim_folder.startswith("Helbing_sim_"):
            result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if result is not None:
                data, config_lines = result
                T = None
                I = None
                for line in config_lines:
                    if "Trail Recovery Rate" in line:
                        T = convert_float(line.split("# Trail Recovery Rate:")[1].strip())
                    elif "Footstep Intensity" in line:
                        I = convert_float(line.split("# Footstep Intensity:")[1].strip())
                
                if T is not None and I is not None:
                    ax.plot(data['Step'], data['Efficiency'], 
                           label=f'T={T:.1f}, I={I:.1f}', alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.set_title('Efficiency over time')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Efficiency')
    ax.set_ylim(0, 4)
    
    # Place legend outside and to the right
    ax.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize='small')

    # Adjust height to match legend
    adjust_plot_height(ax, fig)

    # Save with tight bbox to remove excess whitespace
    plt.savefig(os.path.join(results_folder, 'efficiency_time_series.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

    # Plot efficiency heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(efficiency_matrix, 
        xticklabels=T_values, 
        yticklabels=I_values,
        cmap='viridis')
    plt.title('Final Efficiency')
    plt.xlabel('Trail Recovery Rate (T)')
    plt.ylabel('Footstep Intensity (I)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'efficiency_heatmap.png'))
    plt.close()

    # For civility time series
    fig = plt.figure(figsize=(12, 3))  # Start with a wider, shorter figure
    ax = plt.gca()
    
    for sim_folder in files:
        if sim_folder.startswith("Helbing_sim_"):
            result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if result is not None:
                data, config_lines = result
                T = None
                I = None
                for line in config_lines:
                    if "Trail Recovery Rate" in line:
                        T = convert_float(line.split("# Trail Recovery Rate:")[1].strip())
                    elif "Footstep Intensity" in line:
                        I = convert_float(line.split("# Footstep Intensity:")[1].strip())
                
                if T is not None and I is not None:
                    scaled_civility = data['Civility']
                    ax.plot(data['Step'], scaled_civility, 
                            label=f'T={T:.1f}, I={I:.1f}', alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.set_title('Civility over time')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Civility')
    ax.set_ylim(0, 20)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    # Place legend outside and to the right
    ax.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize='small')

    # Adjust height to match legend
    adjust_plot_height(ax, fig)

    # Save with tight bbox to remove excess whitespace
    plt.savefig(os.path.join(results_folder, 'civility_time_series.png'), 
                bbox_inches='tight',
                dpi=300)
    plt.close()

    # For performance time series
    fig = plt.figure(figsize=(12, 3))
    ax = plt.gca()
    
    for sim_folder in files:
        if sim_folder.startswith("Helbing_sim_"):
            result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if result is not None:
                data, config_lines = result
                T = None
                I = None
                for line in config_lines:
                    if "Trail Recovery Rate" in line:
                        T = convert_float(line.split("# Trail Recovery Rate:")[1].strip())
                    elif "Footstep Intensity" in line:
                        I = convert_float(line.split("# Footstep Intensity:")[1].strip())
                
                if T is not None and I is not None:
                    ax.plot(data['Step'], data['TimeForInterval'], 
                            label=f'T={T:.1f}, I={I:.1f}', alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.set_title('Computation Time per Step')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Time (seconds)')

    # Place legend outside and to the right
    ax.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize='small')

    # Adjust height to match legend
    adjust_plot_height(ax, fig)

    # Save with tight bbox to remove excess whitespace
    plt.savefig(os.path.join(results_folder, 'performance_time_series.png'), 
                bbox_inches='tight',
                dpi=300)
    plt.close()

    # Plot performance heatmap
    plt.figure(figsize=(8, 6))
    performance_matrix = np.zeros((len(I_values), len(T_values)))
    for T, I, eff, civ in results:
        i = I_values.index(I)
        j = T_values.index(T)
        # Use the average of last 100 steps
        result = load_data(os.path.join(test_folder, f"Helbing_sim_{i*len(T_values)+j+1}", "data.csv"))
        if result is not None:
            data, _ = result
            performance_matrix[i,j] = data['TimeForInterval'].iloc[-100:].mean()

    sns.heatmap(performance_matrix,
        xticklabels=T_values,
        yticklabels=I_values,
        cmap='viridis')
    plt.title('Final Computation Time')
    plt.xlabel('Trail Recovery Rate (T)')
    plt.ylabel('Footstep Intensity (I)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance_heatmap.png'))
    plt.close()

    # Plot civility heatmap with Unity colors
    plt.figure(figsize=(8, 6))
    colors = [
        'red',          # comfortColor0 (0)
        'yellow',       # comfortColor1 (~20%)
        'green',        # comfortColor2 (~40%)
        'cyan',         # comfortColor3 (~60%)
        'blue',         # comfortColor4 (~80%)
        'magenta',      # comfortColor5 (~90%)
        '#FF0080'      # comfortColor6 (pink, RGB: 1.0, 0.0, 0.5)
    ]
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    custom_cmap = LinearSegmentedColormap.from_list('comfort_cmap', 
        list(zip(thresholds, colors)))

    # Sort I_values in ascending order and flip the matrix vertically
    I_values = sorted(list(I_values))  # Make sure it's in ascending order
    sns.heatmap(np.flipud(civility_matrix),  # Flip matrix vertically
        xticklabels=T_values, 
        yticklabels=I_values[::-1],  # Reverse the labels to match flipped matrix
        cmap=custom_cmap,
        vmin=0,
        vmax=20)
    plt.title('Final Civility')
    plt.xlabel('Trail Recovery Rate (T)')
    plt.ylabel('Footstep Intensity (I)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'civility_heatmap.png'))
    plt.close()

def calculate_total_samples(vision_arc, first_arc, last_arc):
    """Calculate total sample points with linear interpolation between arcs"""
    vision_arc = int(vision_arc)  # Convert to integer for range()
    
    if vision_arc <= 2:  # Need at least 2 arcs for interpolation
        return vision_arc * first_arc
    
    # For each arc between first and last, calculate interpolated point count
    samples = first_arc + last_arc  # First and last arc
    step = (last_arc - first_arc) / (vision_arc - 1)  # Linear interpolation step
    
    # Add points for each intermediate arc
    for i in range(1, vision_arc - 1):  # Skip first and last arc
        points_on_arc = first_arc + (step * i)
        samples += points_on_arc
        
    return samples

def plot_test3(data_folder):
    """Plot sample points test results"""
    test_folder = os.path.join(data_folder, "Sample_Points")
    results_folder = os.path.join(test_folder, "Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dictionary to store data for each sampling configuration
    sampling_configs = {
        "small": {"arcs": 5, "first": 10, "last": 20, "data": []},
        "medium": {"arcs": 10, "first": 20, "last": 40, "data": []},
        "large": {"arcs": 15, "first": 30, "last": 60, "data": []}
    }
    
    # Collect data for each configuration
    files = os.listdir(test_folder)
    for sim_folder in files:
        if sim_folder.startswith("Vision_sim_"):
            data = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if data is not None:
                data, config_lines = data
                
                # Extract parameters
                vision_arc = first_arc = last_arc = force = sigma = None
                for line in config_lines:
                    if "Vision Arc Count:" in line:
                        vision_arc = convert_float(line.split("Vision Arc Count:")[1].strip())
                    elif "First Arc Point Count:" in line:
                        first_arc = convert_float(line.split("First Arc Point Count:")[1].strip())
                    elif "Last Arc Point Count:" in line:
                        last_arc = convert_float(line.split("Last Arc Point Count:")[1].strip())
                    elif "Visual Path Follow Strength:" in line:
                        force = convert_float(line.split("Visual Path Follow Strength:")[1].strip())
                    elif "Visual Distance Factor:" in line:
                        sigma = convert_float(line.split("Visual Distance Factor:")[1].strip())
                
                # Determine which configuration this belongs to
                for config_name, config in sampling_configs.items():
                    if (vision_arc == config["arcs"] and 
                        first_arc == config["first"] and 
                        last_arc == config["last"]):
                        efficiency = data['Efficiency'].mean()
                        civility = data['Civility'].mean() * 20  # Scale civility to match Unity
                        config["data"].append((force, sigma, efficiency, civility))
    
    # Create heatmaps for each configuration
    metrics = ["efficiency", "civility"]
    
    # Define Unity's color scheme
    colors = [
        'red',          # comfortColor0 (0)
        'yellow',       # comfortColor1 (~20%)
        'green',        # comfortColor2 (~40%)
        'cyan',         # comfortColor3 (~60%)
        'blue',         # comfortColor4 (~80%)
        'magenta',      # comfortColor5 (~90%)
        '#FF0080'      # comfortColor6 (pink, RGB: 1.0, 0.0, 0.5)
    ]
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    comfort_cmap = LinearSegmentedColormap.from_list('comfort_cmap', 
        list(zip(thresholds, colors)))
    
    for config_name, config in sampling_configs.items():
        if not config["data"]:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        forces = sorted(list(set(x[0] for x in config["data"])))
        sigmas = sorted(list(set(x[1] for x in config["data"])))
        
        for metric_idx, metric in enumerate(metrics):
            matrix = np.zeros((len(forces), len(sigmas)))
            for force_idx, force in enumerate(forces):
                for sigma_idx, sigma in enumerate(sigmas):
                    matching_data = [x[2] if metric == "efficiency" else x[3] 
                                   for x in config["data"] if x[0] == force and x[1] == sigma]
                    if matching_data:
                        matrix[force_idx, sigma_idx] = matching_data[0]
            
            ax = axes[metric_idx]
            
            # Use different colormaps for efficiency and civility
            if metric == "efficiency":
                im = ax.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
            else:  # civility
                im = ax.imshow(matrix, cmap=comfort_cmap, aspect='auto', origin='lower',
                             vmin=0, vmax=20)  # Set range for civility
            
            plt.colorbar(im, ax=ax, label=metric.capitalize())
            
            # Set labels
            ax.set_xticks(range(len(sigmas)))
            ax.set_yticks(range(len(forces)))
            ax.set_xticklabels([f'{s:.1f}' for s in sigmas], rotation=45)
            ax.set_yticklabels([f'{f:.1f}' for f in forces])
            
            ax.set_xlabel('Sigma (σ)')
            ax.set_ylabel('Force')
            ax.set_title(f'{metric.capitalize()} Heatmap for {config_name.capitalize()} Sampling\n' +
                        f'(Arcs: {config["arcs"]}, Points: {config["first"]}-{config["last"]})')
            
            # Add text annotations
            for i in range(len(forces)):
                for j in range(len(sigmas)):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="w")
            
            # Find and mark best parameters
            best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
            best_force = forces[best_idx[0]]
            best_sigma = sigmas[best_idx[1]]
            best_value = matrix[best_idx]
            
            # Add text box with best parameters
            ax.text(0.95, 0.95, f'Best Parameters:\nForce: {best_force:.1f}\nSigma: {best_sigma:.1f}\n{metric.capitalize()}: {best_value:.2f}',
                    transform=ax.transAxes, verticalalignment='top',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{config_name}_sampling_comparison.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

def plot_test2(data_folder):
    """Plot path following parameters test results"""
    test_folder = os.path.join(data_folder, "Force_and_Sigma")
    results_folder = os.path.join(test_folder, "Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dictionary to store data for both methods
    methods_data = {
        "helbing": {"data": []},  # Will store (force, sigma, efficiency, civility) tuples
        "vision": {"data": []}
    }
    
    # Collect data
    files = os.listdir(test_folder)
    for sim_folder in files:
        data = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
        if data is not None:
            data, config_lines = data
            force = sigma = None
            
            # Extract parameters based on method
            if sim_folder.startswith("Helbing_sim_"):
                method = "helbing"
                for line in config_lines:
                    if "# Path Follow Strength:" in line and "Visual" not in line:  # More specific match
                        force = convert_float(line.split("# Path Follow Strength:")[1].strip())
                    elif "# Helbing Distance Factor:" in line:
                        sigma = convert_float(line.split("# Helbing Distance Factor:")[1].strip())
            elif sim_folder.startswith("Vision_sim_"):
                method = "vision"
                for line in config_lines:
                    if "# Visual Path Follow Strength:" in line:  # Exact match
                        force = convert_float(line.split("# Visual Path Follow Strength:")[1].strip())
                    elif "# Visual Distance Factor:" in line:
                        sigma = convert_float(line.split("# Visual Distance Factor:")[1].strip())
            
            if force is not None and sigma is not None:
                efficiency = data['Efficiency'].mean()
                civility = data['Civility'].mean() * 20  # Already fixed
                methods_data[method]["data"].append((force, sigma, efficiency, civility))
    
    # After collecting data
    print("\nDebug: Collected data for Helbing method:")
    for data_point in methods_data["helbing"]["data"]:
        print(f"Force: {data_point[0]}, Sigma: {data_point[1]}")
    
    print("\nDebug: Collected data for Vision method:")
    for data_point in methods_data["vision"]["data"]:
        print(f"Force: {data_point[0]}, Sigma: {data_point[1]}")
    
    # Before creating matrix
    print("\nDebug: Unique force values for each method:")
    for method in methods_data:
        data = methods_data[method]["data"]
        forces = sorted(list(set(x[0] for x in data)))
        print(f"{method}: {forces}")
    
    # Create heatmaps for each method and metric
    metrics = ["efficiency", "civility"]
    methods = ["helbing", "vision"]
    
    # Define Unity's color scheme
    colors = [
        'red',          # comfortColor0 (0)
        'yellow',       # comfortColor1 (~20%)
        'green',        # comfortColor2 (~40%)
        'cyan',         # comfortColor3 (~60%)
        'blue',         # comfortColor4 (~80%)
        'magenta',      # comfortColor5 (~90%)
        '#FF0080'      # comfortColor6 (pink, RGB: 1.0, 0.0, 0.5)
    ]
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    comfort_cmap = LinearSegmentedColormap.from_list('comfort_cmap', 
        list(zip(thresholds, colors)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            data = methods_data[method]["data"]
            if not data:
                continue
            
            # Extract unique force and sigma values - sort in ascending order
            forces = sorted(list(set(x[0] for x in data)))  # Removed reverse=True
            sigmas = sorted(list(set(x[1] for x in data)))
            
            # Create matrix
            matrix = np.zeros((len(forces), len(sigmas)))
            for force_idx, force in enumerate(forces):
                for sigma_idx, sigma in enumerate(sigmas):
                    matching_data = [x[2] if metric == "efficiency" else x[3] 
                                   for x in data if x[0] == force and x[1] == sigma]
                    if matching_data:
                        matrix[force_idx, sigma_idx] = matching_data[0]
            
            ax = axes[i, j]
            
            # Use different colormaps for efficiency and civility, but flip the matrix vertically
            if metric == "efficiency":
                im = ax.imshow(np.flipud(matrix), cmap='viridis', aspect='auto', origin='lower')
            else:  # civility
                im = ax.imshow(np.flipud(matrix), cmap=comfort_cmap, aspect='auto', origin='lower',
                             vmin=0, vmax=20)
            
            plt.colorbar(im, ax=ax, label=metric.capitalize())
            
            # Set labels
            ax.set_xticks(range(len(sigmas)))
            ax.set_yticks(range(len(forces)))
            ax.set_xticklabels([f'{s:.1f}' for s in sigmas], rotation=45)
            ax.set_yticklabels([f'{f:.1f}' for f in forces])
            
            ax.set_xlabel('Sigma (σ)')
            ax.set_ylabel('Force')
            method_name = "Helbing's" if method == "helbing" else "Vision-based"
            ax.set_title(f'{method_name} Method - {metric.capitalize()}')
            
            # Add text annotations
            for fi, force in enumerate(forces):
                for si, sigma in enumerate(sigmas):
                    text = ax.text(si, fi, f'{matrix[fi, si]:.2f}',
                                 ha="center", va="center", color="w")
            
            # Find and mark best parameters
            best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
            best_force = forces[best_idx[0]]
            best_sigma = sigmas[best_idx[1]]
            best_value = matrix[best_idx]
            
            # Add text box with best parameters
            ax.text(0.95, 0.95, f'Best Parameters:\nForce: {best_force:.1f}\nSigma: {best_sigma:.1f}\n{metric.capitalize()}: {best_value:.2f}',
                    transform=ax.transAxes, verticalalignment='top',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'method_comparison_heatmaps.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_test4(data_folder):
    """Plot performance scaling test results"""
    test_folder = os.path.join(data_folder, "4_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dictionary to store data for each resolution and method
    data_by_resolution = {
        100: {"helbing": [], "vision": []},
        200: {"helbing": [], "vision": []},
        300: {"helbing": [], "vision": []}
    }
    
    # Collect data
    files = os.listdir(test_folder)
    for sim_folder in files:
        data = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
        if data is not None:
            data, config_lines = data
            resolution = None
            method = None
            agent_count = None
            
            for line in config_lines:
                if "Resolution:" in line:
                    resolution = int(line.split("Resolution:")[1].strip())
                elif "Agent Count:" in line:
                    agent_count = int(line.split("Agent Count:")[1].strip())
            
            if sim_folder.startswith("Helbing"):
                method = "helbing"
            elif sim_folder.startswith("Vision"):
                method = "vision"
            
            if resolution and method and agent_count:
                data_by_resolution[resolution][method].append((agent_count, data))

    # Create time series plots for each resolution
    for resolution in [100, 200, 300]:
        plt.figure(figsize=(15, 8))
        
        # Sort data by agent count
        for method in ["helbing", "vision"]:
            data_by_resolution[resolution][method].sort(key=lambda x: x[0])
            
            # Group data by agent count
            agent_groups = {}
            for agent_count, data in data_by_resolution[resolution][method]:
                if agent_count not in agent_groups:
                    agent_groups[agent_count] = []
                agent_groups[agent_count].append(data)
            
            # Plot each agent count
            for agent_count, group_data in agent_groups.items():
                # Calculate mean and std for each time step
                mean_times = np.mean([data['TimeForInterval'] for data in group_data], axis=0)
                std_times = np.std([data['TimeForInterval'] for data in group_data], axis=0)
                steps = np.arange(len(mean_times)) * 100  # Convert to actual step numbers
                
                plt.plot(steps, mean_times, 
                        label=f"{'Helbing' if method == 'helbing' else 'Vision'}-based ({agent_count} agents)",
                        marker='o', markersize=4)
                plt.fill_between(steps, 
                               mean_times - std_times, 
                               mean_times + std_times, 
                               alpha=0.2)
        
        plt.title(f'Performance Over Time at Resolution {resolution}×{resolution}')
        plt.xlabel('Simulation Step')
        plt.ylabel('Time per Step (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'performance_time_series_res{resolution}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    # Create bar plots for each resolution
    for resolution in [100, 200, 300]:
        plt.figure(figsize=(12, 6))
        
        agent_counts = sorted(list(set(x[0] for x in data_by_resolution[resolution]["helbing"])))
        x = np.arange(len(agent_counts))
        width = 0.35
        
        # Calculate means and standard deviations
        helbing_means = []
        helbing_stds = []
        vision_means = []
        vision_stds = []
        
        for ac in agent_counts:
            helbing_data = [d[1]['TimeForInterval'].mean() for d in data_by_resolution[resolution]["helbing"] if d[0] == ac]
            vision_data = [d[1]['TimeForInterval'].mean() for d in data_by_resolution[resolution]["vision"] if d[0] == ac]
            
            helbing_means.append(np.mean(helbing_data))
            helbing_stds.append(np.std(helbing_data))
            vision_means.append(np.mean(vision_data))
            vision_stds.append(np.std(vision_data))
        
        plt.bar(x - width/2, helbing_means, width, yerr=helbing_stds, 
                label="Helbing's Method", capsize=5)
        plt.bar(x + width/2, vision_means, width, yerr=vision_stds, 
                label="Vision-based Method", capsize=5)
        
        plt.title(f'Average Performance at Resolution {resolution}×{resolution}')
        plt.xlabel('Number of Agents')
        plt.ylabel('Average Time per Step (seconds)')
        plt.xticks(x, agent_counts)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'performance_bar_res{resolution}.png'))
        plt.close()

    # Create combined plot for all resolutions
    plt.figure(figsize=(15, 8))
    
    for resolution in [100, 200, 300]:
        for method in ["helbing", "vision"]:
            data_by_resolution[resolution][method].sort(key=lambda x: x[0])
            
            # Group data by agent count
            agent_groups = {}
            for agent_count, data in data_by_resolution[resolution][method]:
                if agent_count not in agent_groups:
                    agent_groups[agent_count] = []
                agent_groups[agent_count].append(data)
            
            # Plot each agent count
            for agent_count, group_data in agent_groups.items():
                mean_times = np.mean([data['TimeForInterval'] for data in group_data], axis=0)
                std_times = np.std([data['TimeForInterval'] for data in group_data], axis=0)
                steps = np.arange(len(mean_times)) * 100
                
                plt.plot(steps, mean_times, 
                        label=f"{'Helbing' if method == 'helbing' else 'Vision'}-based ({resolution}×{resolution}, {agent_count} agents)",
                        marker='o', markersize=4)
                plt.fill_between(steps, 
                               mean_times - std_times, 
                               mean_times + std_times, 
                               alpha=0.2)
    
    plt.title('Performance Scaling Across All Configurations')
    plt.xlabel('Simulation Step')
    plt.ylabel('Time per Step (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance_combined.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Create combined bar plot
    plt.figure(figsize=(15, 8))
    
    # Prepare data for grouped bar chart
    resolutions = [100, 200, 300]
    agent_counts = sorted(list(set(x[0] for x in data_by_resolution[100]["helbing"])))
    x = np.arange(len(agent_counts))
    width = 0.1  # Width of each bar
    
    for i, resolution in enumerate(resolutions):
        for method in ["helbing", "vision"]:
            means = []
            stds = []
            for ac in agent_counts:
                data = [d[1]['TimeForInterval'].mean() for d in data_by_resolution[resolution][method] if d[0] == ac]
                means.append(np.mean(data))
                stds.append(np.std(data))
            
            offset = (i * 2 + (0 if method == "helbing" else 1)) * width
            plt.bar(x + offset, means, width, yerr=stds, capsize=5,
                   label=f"{'Helbing' if method == 'helbing' else 'Vision'}-based ({resolution}×{resolution})")
    
    plt.title('Average Performance Across All Configurations')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Time per Step (seconds)')
    plt.xticks(x + width * 2, agent_counts)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance_combined_bar.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Look for data in Unity's default location
    data_folder = "../ExperimentData"
    if not os.path.exists(data_folder):
        print(f"Creating data folder: {os.path.abspath(data_folder)}")
        os.makedirs(data_folder)
    
    #plot_test1(data_folder)
    #plot_test2(data_folder)
    #plot_test3(data_folder)
    plot_test4(data_folder) 