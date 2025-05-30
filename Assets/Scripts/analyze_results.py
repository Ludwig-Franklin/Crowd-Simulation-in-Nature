import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Fix for colormap
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

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
    
    # Define Unity's comfort colormap
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
    comfort_cmap = LinearSegmentedColormap.from_list('comfort_cmap', list(zip(thresholds, colors)))
    
    # Store T and I values for each simulation
    T_values = set()
    I_values = set()
    results = []
    
    # Collect data and find final screenshots
    files = os.listdir(test_folder)
    for sim_folder in files:
        if sim_folder.startswith("Helbing_sim_"):
            result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if result is not None:
                data, config_lines = result
                T = None
                I = None
                for line in config_lines:
                    if "Trail Recovery Rate" in line:
                        T = convert_float(line.split("Trail Recovery Rate:")[1].strip())
                    elif "Footstep Intensity" in line:
                        I = convert_float(line.split("Footstep Intensity:")[1].strip())
                
                if T is not None and I is not None:
                    T_values.add(T)
                    I_values.add(I)
                    final_efficiency = data['Efficiency'].iloc[-1]
                    final_civility = data['Civility'].iloc[-1] * 20
                    results.append((T, I, final_efficiency, final_civility))
    
    # Create screenshot grid
    T_values = sorted(list(T_values))
    I_values = sorted(list(I_values))
    
    # Remove first row and column (they are empty)
    T_values = T_values[1:]
    I_values = I_values[1:]
    
    # Make plots bigger and tighter together
    fig, axes = plt.subplots(len(I_values), len(T_values), 
                            figsize=(5*len(T_values), 5*len(I_values)),
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    
    # Make axes 2D if it's 1D
    if len(I_values) == 1:
        axes = np.array([axes])
    if len(T_values) == 1:
        axes = axes.reshape(-1, 1)
    
    # Find matching screenshots for each T,I combination
    for i, I in enumerate(I_values):
        for j, T in enumerate(T_values):
            found_screenshot = False
            # Look through all simulation folders for matching T and I values
            for sim_num in range(1, 50):  # Check Helbing_sim_1 through Helbing_sim_49
                sim_folder = f"Helbing_sim_{sim_num}"
                img_path = os.path.join(test_folder, sim_folder, "step_1999.png")
                
                if os.path.exists(img_path):
                    # Check if this simulation matches our T and I values
                    result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
                    if result is not None:
                        data, config_lines = result
                        sim_T = None
                        sim_I = None
                        for line in config_lines:
                            if "Trail Recovery Rate" in line:
                                sim_T = convert_float(line.split("Trail Recovery Rate:")[1].strip())
                            elif "Footstep Intensity" in line:
                                sim_I = convert_float(line.split("Footstep Intensity:")[1].strip())
                        
                        if sim_T == T and sim_I == I:
                            # Load and crop image
                            img = plt.imread(img_path)
                            h, w = img.shape[:2]
                            crop_left = int(w * 0.25)
                            crop_right = int(w * 0.75)
                            crop_top = int(h * 0.1)
                            crop_bottom = int(h * 0.9)
                            cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]
                            
                            axes[i,j].imshow(cropped_img)
                            found_screenshot = True
                            break
            
            if found_screenshot:
                axes[i,j].set_title(f'T={T}, I={I}', fontsize=14, fontweight='bold')
                axes[i,j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'final_trails_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create time series plot for efficiency and civility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    plt.subplots_adjust(hspace=0.3)  # Add space between plots
    
    for sim_folder in files:
        if sim_folder.startswith("Helbing_sim_"):
            result = load_data(os.path.join(test_folder, sim_folder, "data.csv"))
            if result is not None:
                data, config_lines = result
                T = None
                I = None
                for line in config_lines:
                    if "Trail Recovery Rate" in line:
                        T = convert_float(line.split("Trail Recovery Rate:")[1].strip())
                    elif "Footstep Intensity" in line:
                        I = convert_float(line.split("Footstep Intensity:")[1].strip())
                
                if T is not None and I is not None:
                    label = f'T={T}, I={I}'
                    ax1.plot(data['Step'], data['Efficiency'], label=label, alpha=0.7)
                    ax2.plot(data['Step'], data['Civility'] * 20, alpha=0.7)  # Scale civility to 0-20
    
    # Configure efficiency subplot
    ax1.set_title('Efficiency over Time')
    ax1.set_ylabel('Efficiency')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure civility subplot
    ax2.set_title('Civility over Time')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Civility')
    ax2.set_ylim(0, 20)  # Set y-axis limits for civility
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(results_folder, 'metrics_time_series.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Create matrices for heatmaps
    T_values = sorted(list(T_values))
    I_values = sorted(list(I_values))
    
    # Remove first row and column (they are empty)
    T_values = T_values[1:]
    I_values = I_values[1:]
    
    # Filter results to only include values that exist in our trimmed T_values and I_values lists
    filtered_results = [(T, I, eff, civ) for T, I, eff, civ in results if T in T_values and I in I_values]
    
    efficiency_matrix = np.zeros((len(I_values), len(T_values)))
    civility_matrix = np.zeros((len(I_values), len(T_values)))
    
    for T, I, eff, civ in filtered_results:
        i = I_values.index(I)
        j = T_values.index(T)
        efficiency_matrix[i, j] = eff
        civility_matrix[i, j] = civ
    
    # Create combined heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot efficiency
    sns.heatmap(efficiency_matrix,
                xticklabels=T_values,
                yticklabels=I_values,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                ax=axes[0],
                cbar_kws={'label': 'Efficiency'})
    axes[0].set_title('Final Efficiency')
    axes[0].set_xlabel('Trail Recovery Rate (T)')
    axes[0].set_ylabel('Footstep Intensity (I)')
    
    # Plot civility with Unity colors
    sns.heatmap(civility_matrix,
                xticklabels=T_values,
                yticklabels=I_values,
                cmap=comfort_cmap,
                vmin=0,
                vmax=20,
                annot=True,
                fmt='.2f',
                ax=axes[1],
                cbar_kws={'label': 'Civility'})
    axes[1].set_title('Final Civility')
    axes[1].set_xlabel('Trail Recovery Rate (T)')
    axes[1].set_ylabel('Footstep Intensity (I)')
    
    plt.suptitle('Trail Formation Parameter Effects', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'combined_metrics_heatmap.png'),
                bbox_inches='tight', dpi=300)
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
    test_folder = os.path.join(data_folder, "3_Sample_points_paired_with_forces_and_sigmas_small")
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Define Unity's color scheme for comfort map
    colors = [
        'red', 'yellow', 'green', 'cyan', 'blue', 'magenta', '#FF0080'
    ]
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    comfort_cmap = LinearSegmentedColormap.from_list('comfort_cmap', 
        list(zip(thresholds, colors)))
    
    # Dictionary to store data for each sampling configuration
    sampling_configs = {
        "small": {"arcs": 5, "first": 10, "last": 20, "data": [], "row": 0},
        "medium": {"arcs": 10, "first": 20, "last": 40, "data": [], "row": 1},
        "large": {"arcs": 15, "first": 30, "last": 60, "data": [], "row": 2}
    }
    
    # First collect all data and screenshots
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
                
                # Find matching configuration
                for config_name, config in sampling_configs.items():
                    if (vision_arc == config["arcs"] and 
                        first_arc == config["first"] and 
                        last_arc == config["last"]):
                        efficiency = data['Efficiency'].mean()
                        civility = data['Civility'].mean() * 20
                        
                        # Also store the screenshot path if it exists
                        screenshot_path = os.path.join(test_folder, sim_folder, "step_1999.png")
                        if os.path.exists(screenshot_path):
                            config["data"].append((force, sigma, efficiency, civility, screenshot_path))
    
    # FIGURE 1: Stacked Heatmaps
    fig_heatmaps = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3)
    
    metrics = ["efficiency", "civility"]
    for config_name, config in sampling_configs.items():
        forces = sorted(list(set(x[0] for x in config["data"])))
        sigmas = sorted(list(set(x[1] for x in config["data"])))
        
        ax1 = fig_heatmaps.add_subplot(gs[config["row"], 0])
        ax2 = fig_heatmaps.add_subplot(gs[config["row"], 1])
        
        for metric_idx, (metric, ax) in enumerate(zip(metrics, [ax1, ax2])):
            matrix = np.zeros((len(forces), len(sigmas)))
            for force_idx, force in enumerate(forces):
                for sigma_idx, sigma in enumerate(sigmas):
                    matching_data = [x[2] if metric == "efficiency" else x[3] 
                                   for x in config["data"] if x[0] == force and x[1] == sigma]
                    if matching_data:
                        matrix[force_idx, sigma_idx] = matching_data[0]
            
            # Use seaborn's heatmap instead of imshow
            if metric == "efficiency":
                sns.heatmap(matrix,
                           xticklabels=[f'{s:.1f}' for s in sigmas],
                           yticklabels=[f'{f:.1f}' for f in forces],
                           cmap='viridis',
                           annot=True,
                           fmt='.2f',
                           ax=ax,
                           cbar_kws={'label': metric.capitalize()})
            else:
                sns.heatmap(matrix,
                           xticklabels=[f'{s:.1f}' for s in sigmas],
                           yticklabels=[f'{f:.1f}' for f in forces],
                           cmap=comfort_cmap,
                           vmin=0,
                           vmax=20,
                           annot=True,
                           fmt='.2f',
                           ax=ax,
                           cbar_kws={'label': metric.capitalize()})
            
            ax.set_xlabel('Sigma (σ)')
            ax.set_ylabel('Force')
            ax.set_title(f'{metric.capitalize()} for {config_name.capitalize()} Sampling\n' +
                        f'(Arcs: {config["arcs"]}, Points: {config["first"]}-{config["last"]})')

    plt.suptitle('Sample Points Configuration Comparison', y=0.95, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'sampling_comparison_heatmaps.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # FIGURE 2: Screenshots Grid
    fig_screenshots = plt.figure(figsize=(20, 25))
    
    # Create a 2x1 grid (Helbing on top, Vision on bottom)
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Process each method
    for method_idx, method in enumerate(["helbing", "vision"]):
        ax = fig_screenshots.add_subplot(gs[method_idx])
        ax.axis('off')
        
        # Get the data for this method
        data = methods_data[method]["data"]
        forces = sorted(list(set(x[0] for x in data)))
        sigmas = sorted(list(set(x[1] for x in data)))
        
        # Create subplot grid
        n_rows = len(forces)  # One row per force value
        n_cols = len(sigmas)  # One column per sigma value
        
        # For each force-sigma combination
        for force_idx, force in enumerate(forces):
            for sigma_idx, sigma in enumerate(sigmas):
                # Find the corresponding simulation folder
                sim_prefix = "Helbing_sim_" if method == "helbing" else "Vision_sim_"
                screenshot_path = None
                
                # Search for matching simulation
                for sim_folder in os.listdir(test_folder):
                    if sim_folder.startswith(sim_prefix):
                        data_file = os.path.join(test_folder, sim_folder, "data.csv")
                        if os.path.exists(data_file):
                            result = load_data(data_file)
                            if result is not None:
                                _, config_lines = result
                                sim_force = sim_sigma = None
                                
                                # Extract parameters based on method
                                for line in config_lines:
                                    if method == "helbing":
                                        if "Path Follow Strength:" in line and "Visual" not in line:
                                            sim_force = convert_float(line.split("Path Follow Strength:")[1].strip())
                                        elif "Helbing Distance Factor:" in line:
                                            sim_sigma = convert_float(line.split("Helbing Distance Factor:")[1].strip())
                                    else:  # vision
                                        if "Visual Path Follow Strength:" in line:
                                            sim_force = convert_float(line.split("Visual Path Follow Strength:")[1].strip())
                                        elif "Visual Distance Factor:" in line:
                                            sim_sigma = convert_float(line.split("Visual Distance Factor:")[1].strip())
                                
                                if abs(sim_force - force) < 0.1 and abs(sim_sigma - sigma) < 0.1:
                                    screenshot_path = os.path.join(test_folder, sim_folder, "step_1999.png")
                                    break
                
                if screenshot_path and os.path.exists(screenshot_path):
                    # Calculate position for this screenshot
                    left = sigma_idx / n_cols
                    bottom = 1 - (force_idx + 1) / n_rows
                    width = 0.9 / n_cols
                    height = 0.9 / n_rows
                    
                    ax_sub = ax.inset_axes([left, bottom, width, height])
                    img = plt.imread(screenshot_path)
                    
                    # Crop the image to focus on the relevant part
                    h, w = img.shape[:2]
                    crop_left = int(w * 0.25)
                    crop_right = int(w * 0.75)
                    crop_top = int(h * 0.1)
                    crop_bottom = int(h * 0.9)
                    screenshot = img[crop_top:crop_bottom, crop_left:crop_right]
                    
                    ax_sub.imshow(screenshot)
                    ax_sub.set_title(f'F={force:.1f}, σ={sigma:.1f}', fontsize=8, pad=3)
                    ax_sub.axis('off')
        
        # Add method label
        method_name = "Helbing's Method" if method == "helbing" else "Vision-based Method"
        ax.set_title(method_name, pad=20, fontsize=14, fontweight='bold')
        
        # Add axis labels
        ax.text(-0.05, 0.5, 'Force', rotation=90, 
                transform=ax.transAxes, va='center', ha='right', fontsize=12)
        ax.text(0.5, -0.05, 'Sigma (σ)', 
                transform=ax.transAxes, va='top', ha='center', fontsize=12)

    #plt.suptitle('Force and Sigma Parameter Comparison - Final States', y=0.95, fontsize=16)
    plt.savefig(os.path.join(results_folder, 'force_sigma_screenshots_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test2(data_folder):
    """Plot path following parameters test results"""
    test_folder = os.path.join(data_folder, "Force_and_Sigma")
    results_folder = os.path.join(test_folder, "Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dictionary to store data for both methods
    methods_data = {
        "helbing": {"data": []},
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
                    if "Path Follow Strength:" in line and "Visual" not in line:
                        force = convert_float(line.split("Path Follow Strength:")[1].strip())
                    elif "Helbing Distance Factor:" in line:
                        sigma = convert_float(line.split("Helbing Distance Factor:")[1].strip())
            elif sim_folder.startswith("Vision_sim_"):
                method = "vision"
                for line in config_lines:
                    if "Visual Path Follow Strength:" in line:
                        force = convert_float(line.split("Visual Path Follow Strength:")[1].strip())
                    elif "Visual Distance Factor:" in line:
                        sigma = convert_float(line.split("Visual Distance Factor:")[1].strip())
            
            if force is not None and sigma is not None:
                efficiency = data['Efficiency'].mean()
                civility = data['Civility'].mean() * 20
                screenshot_path = os.path.join(test_folder, sim_folder, "step_1999.png")
                if os.path.exists(screenshot_path):
                    methods_data[method]["data"].append((force, sigma, efficiency, civility, screenshot_path))
    
    # FIGURE 1: Method Comparison Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    metrics = ["efficiency", "civility"]
    methods = ["helbing", "vision"]
    
    # Define Unity's color scheme
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', '#FF0080']
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    comfort_cmap = LinearSegmentedColormap.from_list('comfort_cmap', list(zip(thresholds, colors)))
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            data = methods_data[method]["data"]
            if not data:
                continue
            
            forces = sorted(list(set(x[0] for x in data)))
            sigmas = sorted(list(set(x[1] for x in data)))
            
            matrix = np.zeros((len(forces), len(sigmas)))
            for force_idx, force in enumerate(forces):
                for sigma_idx, sigma in enumerate(sigmas):
                    matching_data = [x[2] if metric == "efficiency" else x[3] 
                                   for x in data if x[0] == force and x[1] == sigma]
                    if matching_data:
                        matrix[force_idx, sigma_idx] = matching_data[0]
            
            ax = axes[i, j]
            
            # Use seaborn's heatmap
            if metric == "efficiency":
                sns.heatmap(matrix,
                           xticklabels=[f'{s:.1f}' for s in sigmas],
                           yticklabels=[f'{f:.1f}' for f in forces],
                           cmap='viridis',
                           annot=True,
                           fmt='.2f',
                           ax=ax,
                           cbar_kws={'label': metric.capitalize()})
            else:
                sns.heatmap(matrix,
                           xticklabels=[f'{s:.1f}' for s in sigmas],
                           yticklabels=[f'{f:.1f}' for f in forces],
                           cmap=comfort_cmap,
                           vmin=0,
                           vmax=20,
                           annot=True,
                           fmt='.2f',
                           ax=ax,
                           cbar_kws={'label': metric.capitalize()})
            
            ax.set_xlabel('Sigma (σ)')
            ax.set_ylabel('Force')
            method_name = "Helbing's" if method == "helbing" else "Vision-based"
            ax.set_title(f'{method_name} Method - {metric.capitalize()}')
    
    #plt.suptitle('Force and Sigma Parameter Comparison', y=0.95, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'force_sigma_heatmaps.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # FIGURE 2: Screenshots Grid
    fig_screenshots = plt.figure(figsize=(20, 25))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    for method_idx, method in enumerate(["helbing", "vision"]):
        ax = fig_screenshots.add_subplot(gs[method_idx])
        ax.axis('off')
        
        data = methods_data[method]["data"]
        forces = sorted(list(set(x[0] for x in data)))
        sigmas = sorted(list(set(x[1] for x in data)))
        
        n_rows = len(forces)
        n_cols = len(sigmas)
        
        for force_idx, force in enumerate(forces):
            for sigma_idx, sigma in enumerate(sigmas):
                matching_data = [x for x in data if x[0] == force and x[1] == sigma]
                if matching_data:
                    screenshot_path = matching_data[0][4]
                    
                    left = sigma_idx / n_cols
                    bottom = 1 - (force_idx + 1) / n_rows
                    width = 0.9 / n_cols
                    height = 0.9 / n_rows
                    
                    ax_sub = ax.inset_axes([left, bottom, width, height])
                    img = plt.imread(screenshot_path)
                    
                    h, w = img.shape[:2]
                    crop_left = int(w * 0.25)
                    crop_right = int(w * 0.75)
                    crop_top = int(h * 0.1)
                    crop_bottom = int(h * 0.9)
                    screenshot = img[crop_top:crop_bottom, crop_left:crop_right]
                    
                    ax_sub.imshow(screenshot)
                    ax_sub.set_title(f'F={force:.1f}, σ={sigma:.1f}', fontsize=8, pad=3)
                    ax_sub.axis('off')
        
        method_name = "Helbing's Method" if method == "helbing" else "Vision-based Method"
        ax.set_title(method_name, pad=20, fontsize=14, fontweight='bold')
        
        ax.text(-0.05, 0.5, 'Force', rotation=90, 
                transform=ax.transAxes, va='center', ha='right', fontsize=12)
        ax.text(0.5, -0.05, 'Sigma (σ)', 
                transform=ax.transAxes, va='top', ha='center', fontsize=12)
    
    #plt.suptitle('Force and Sigma Parameter Comparison - Final States', y=0.95, fontsize=16)
    plt.savefig(os.path.join(results_folder, 'force_sigma_screenshots_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test4(data_folder):
    """Plot performance scaling test results"""
    test_folder = os.path.join(data_folder, "4_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "_Results")
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

    # Create combined time series plots for each metric, stacked by resolution
    metrics = ['Efficiency', 'Civility', 'TimeForInterval']
    metric_titles = ['Efficiency', 'Civility', 'Time per Step (seconds)']
    
    for metric, title in zip(metrics, metric_titles):
        # Create figure with 3 subplots stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        for res_idx, resolution in enumerate([100, 200, 300]):
            ax = axes[res_idx]
            
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
                    if metric == 'TimeForInterval':
                        # Divide time values by 100 to get actual time per step
                        mean_values = np.mean([data[metric] / 100.0 for data in group_data], axis=0)
                        std_values = np.std([data[metric] / 100.0 for data in group_data], axis=0)
                    else:
                        # Keep other metrics as they are
                        mean_values = np.mean([data[metric] for data in group_data], axis=0)
                        std_values = np.std([data[metric] for data in group_data], axis=0)
                    
                    steps = np.arange(len(mean_values)) * 100
                    
                    ax.plot(steps, mean_values, 
                           label=f"{'Helbing' if method == 'helbing' else 'Vision'}-based ({agent_count} agents)",
                           marker='o', markersize=4)
                    ax.fill_between(steps, 
                                  mean_values - std_values, 
                                  mean_values + std_values, 
                                  alpha=0.2)
            
            ax.set_title(f'Resolution {resolution}×{resolution}')
            ax.set_xlabel('Simulation Step')
            ax.set_ylabel('Time per Step (seconds)' if metric == 'TimeForInterval' else title)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(f'{title} Over Time', y=0.95, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{metric.lower()}_combined_time_series.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

    # Add overall performance comparison bar chart
    plt.figure(figsize=(10, 6))
    
    # Calculate average metrics across all configurations
    metrics = ['Efficiency', 'Civility']
    helbing_means = {metric: [] for metric in metrics}
    helbing_stds = {metric: [] for metric in metrics}
    vision_means = {metric: [] for metric in metrics}
    vision_stds = {metric: [] for metric in metrics}
    
    # Collect all data for each method and metric
    for resolution in [100, 200, 300]:
        for method in ["helbing", "vision"]:
            for metric in metrics:
                all_values = []
                for agent_count, data in data_by_resolution[resolution][method]:
                    all_values.append(data[metric].mean())
                
                if method == "helbing":
                    helbing_means[metric].append(np.mean(all_values))
                    helbing_stds[metric].append(np.std(all_values))
                else:
                    vision_means[metric].append(np.mean(all_values))
                    vision_stds[metric].append(np.std(all_values))
    
    # Create grouped bar chart
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, 
           [np.mean(helbing_means[m]) for m in metrics],
           width,
           yerr=[np.mean(helbing_stds[m]) for m in metrics],
           label="Helbing's Method",
           capsize=5)
    
    plt.bar(x + width/2,
           [np.mean(vision_means[m]) for m in metrics],
           width,
           yerr=[np.mean(vision_stds[m]) for m in metrics],
           label="Vision-based Method",
           capsize=5)
    
    plt.ylabel('Average Value')
    plt.title('Overall Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'overall_performance_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_screenshot_grid(data_folder):
    """Create a grid of screenshots comparing Helbing and Vision methods"""
    test_folder = os.path.join(data_folder, "4_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Configurations to compare
    configs = [
        (100, 30), (100, 40),
        (200, 30), (200, 40)
    ]
    
    # Modified steps to skip even hundreds
    steps = [100, 300, 500, 700, 900, 1000]
    n_steps = len(steps)
    n_rows = len(configs) * 2  # Helbing and Vision for each config
    n_cols = 6  # Reduced to 6 screenshots per row
    
    # Create figure
    fig = plt.figure(figsize=(20, 2.5 * n_rows))
    
    # Add step numbers at the top - centered over each column
    for col_idx, step in enumerate(steps):
        x = 0.1 + (col_idx + 0.5) * 0.85/n_cols
        plt.figtext(x, 
                    0.98, 
                   f'Step {step}',
                   ha='center', va='bottom', fontsize=12, weight='bold')

    for config_idx, (resolution, agent_count) in enumerate(configs):
        row_base = config_idx * 2  # Keep this as integer for subplot indexing
        row_spacing = config_idx * 1.8125  # Use this for text and line positioning
        
        # Calculate y positions for text - adjusted for better centering
        y_helbing = 1 - (row_spacing + 0.875) / n_rows
        y_config_line1 = 1 - (row_spacing + 1.25) / n_rows
        y_config_line2 = 1 - (row_spacing + 1.35) / n_rows
        y_vision = 1 - (row_spacing + 1.75) / n_rows
        
        # Add method labels and configuration text
        plt.figtext(0.02, y_helbing, "Helbing", ha='left', va='center', fontsize=12, weight='bold')
        plt.figtext(0.02, y_config_line1, f'{resolution}×{resolution}', ha='left', va='center', fontsize=12, weight='bold')
        plt.figtext(0.02, y_config_line2, f'{agent_count} agents', ha='left', va='center', fontsize=12, weight='bold')
        plt.figtext(0.02, y_vision, "Vision", ha='left', va='center', fontsize=12, weight='bold')
        
        for method_idx in range(2):
            row_idx = row_base + method_idx  # Use integer row_base for subplot indexing
            
            # Find the corresponding simulation folder
            sim_folder_pattern = f"{'Helbing' if method_idx == 0 else 'Vision'}_resolution={resolution}_agents={agent_count}_repetition=1"
            sim_folders = [f for f in os.listdir(test_folder) 
                         if os.path.isdir(os.path.join(test_folder, f)) 
                         and f.startswith(sim_folder_pattern)]
            
            if not sim_folders:
                print(f"No folder found for {sim_folder_pattern}")
                continue
                
            sim_folder = sim_folders[0]
            screenshot_folder = os.path.join(test_folder, sim_folder)
            
            # Process screenshots for this configuration
            for col_idx, step in enumerate(steps):
                screenshot_file = f'step_{step}.png'
                if os.path.exists(os.path.join(screenshot_folder, screenshot_file)):
                    # Read and crop image
                    img = plt.imread(os.path.join(screenshot_folder, screenshot_file))
                    h, w = img.shape[:2]
                    crop_left = int(w * 0.2)
                    crop_right = int(w * 0.8)
                    crop_top = int(h * 0.1)
                    crop_bottom = int(h * 0.9)
                    cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]
                    
                    # Add to grid
                    ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
                    ax.imshow(cropped_img)
                    ax.axis('off')
            
            # Add horizontal line after Vision method
            if method_idx == 1 and config_idx < len(configs) - 1:
                fig.patches.extend([
                    plt.Rectangle(
                        (0, 1 - (row_spacing + 2.2) / n_rows),  # Use row_spacing for line positioning
                        1,
                        0.002,
                        facecolor='black',
                        transform=fig.transFigure,
                        zorder=10,
                        clip_on=False
                    )
                ])
    
    # Adjust spacing - different spacing within and between groups
    plt.subplots_adjust(
        hspace=0.05,     # Very tight spacing within groups
        wspace=0.05,     # Keep minimal horizontal spacing
        top=0.95,        # Keep top margin
        bottom=0.05,     # Keep bottom margin
        left=0.1,        # Keep left margin
        right=0.95       # Keep right margin
    )
    plt.savefig(os.path.join(results_folder, 'method_comparison_grid.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_combined_heatmap(data_folder):
    """Create a combined heatmap showing Civility and Efficiency for all configurations"""
    test_folder = os.path.join(data_folder, "4_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dictionary to store data
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

    # Create figure with 2x3 subplot grid (two metrics, three columns)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Define Unity's comfort colors at the start of the function
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

    # Process each metric separately
    for row, metric in enumerate(['Civility', 'Efficiency']):
        # Prepare data
        resolutions = [100, 200, 300]
        agent_counts = sorted(list(set(x[0] for x in data_by_resolution[100]["helbing"])))
        
        # Create matrices for both methods
        helbing_matrix = np.zeros((len(resolutions), len(agent_counts)))
        vision_matrix = np.zeros((len(resolutions), len(agent_counts)))
        
        # Fill matrices
        for i, res in enumerate(resolutions):
            for j, ac in enumerate(agent_counts):
                helbing_data = [d[1][metric].mean() 
                              for d in data_by_resolution[res]["helbing"] 
                              if d[0] == ac]
                vision_data = [d[1][metric].mean() 
                             for d in data_by_resolution[res]["vision"] 
                             if d[0] == ac]
                
                if helbing_data:
                    helbing_matrix[i, j] = np.mean(helbing_data)
                if vision_data:
                    vision_matrix[i, j] = np.mean(vision_data)
        
        # Calculate difference matrix
        diff_matrix = helbing_matrix - vision_matrix
        
        # Create heatmaps - moved outside the resolution loop
        if metric == 'Civility':
            sns.heatmap(helbing_matrix,
                       xticklabels=agent_counts,
                       yticklabels=[f'{r}×{r}' for r in resolutions],
                       cmap=comfort_cmap,
                       annot=True,
                       fmt='.2f',
                       ax=axes[row,0],
                       vmin=0,
                       vmax=20,
                       cbar_kws={'label': metric})
            
            sns.heatmap(vision_matrix,
                       xticklabels=agent_counts,
                       yticklabels=[f'{r}×{r}' for r in resolutions],
                       cmap=comfort_cmap,
                       annot=True,
                       fmt='.2f',
                       ax=axes[row,1],
                       vmin=0,
                       vmax=20,
                       cbar_kws={'label': metric})
        else:
            # For Efficiency, one colorbar per subplot
            sns.heatmap(helbing_matrix,
                       xticklabels=agent_counts,
                       yticklabels=[f'{r}×{r}' for r in resolutions],
                       cmap='viridis',
                       annot=True,
                       fmt='.2f',
                       ax=axes[row,0],
                       cbar_kws={'label': metric})
            
            sns.heatmap(vision_matrix,
                       xticklabels=agent_counts,
                       yticklabels=[f'{r}×{r}' for r in resolutions],
                       cmap='viridis',
                       annot=True,
                       fmt='.2f',
                       ax=axes[row,1],
                       cbar_kws={'label': metric})
        
        # Difference plot with its own colorbar
        abs_max = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
        sns.heatmap(diff_matrix,
                   xticklabels=agent_counts,
                   yticklabels=[f'{r}×{r}' for r in resolutions],
                   cmap='RdBu',
                   annot=True,
                   fmt='.2f',
                   ax=axes[row,2],
                   center=0,
                   vmin=-abs_max,
                   vmax=abs_max,
                   cbar_kws={'label': 'Difference'})

    plt.suptitle('Method Comparison Across Metrics', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'combined_metrics_heatmap.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_performance_comparison(data_folder):
    """Create a bar chart comparing time performance across all configurations"""
    test_folder = os.path.join(data_folder, "4_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "_Results")
    
    # Dictionary to store data
    data_by_resolution = {
        100: {"helbing": [], "vision": []},
        200: {"helbing": [], "vision": []},
        300: {"helbing": [], "vision": []}
    }
    
    # Collect data (same as before)
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

    # Create bar chart
    plt.figure(figsize=(15, 6))
    
    # Get all agent counts
    agent_counts = sorted(list(set(
        ac for res in data_by_resolution.values() 
        for method in res.values() 
        for ac, _ in method
    )))
    
    # Setup x-axis positions
    n_groups = len(agent_counts)
    n_bars = 6  # 3 resolutions × 2 methods
    bar_width = 0.8 / n_bars
    
    # Colors for each resolution
    colors = {'helbing': ['#1f77b4', '#2ca02c', '#ff7f0e'],  # Blue, Green, Orange
             'vision': ['#17becf', '#98df8a', '#ffbb78']}    # Light versions
    
    # Plot bars
    for i, agent_count in enumerate(agent_counts):
        x_center = i
        
        for j, (resolution, color_idx) in enumerate(zip([100, 200, 300], range(3))):
            for k, method in enumerate(['helbing', 'vision']):
                # Calculate bar position
                x_pos = x_center - 0.4 + (j*2 + k) * bar_width
                
                # Get data and divide by 100 to get actual time per step
                times = [data['TimeForInterval'].mean() / 100.0  # Divide by 100
                        for ac, data in data_by_resolution[resolution][method] 
                        if ac == agent_count]
                
                if times:
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    
                    plt.bar(x_pos, mean_time, bar_width,
                           yerr=std_time, capsize=3,
                           color=colors[method][color_idx],
                           label=f'{resolution}×{resolution} {method.capitalize()}'
                           if i == 0 else None)

    plt.xlabel('Number of Agents')
    plt.ylabel('Time per Step (seconds)')  # Updated label
    plt.title('Performance Comparison Across All Configurations')
    plt.xticks(range(len(agent_counts)), agent_counts)
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_combined_time_series(data_folder):
    """Create combined time series plots for each metric, stacked by resolution"""
    test_folder = os.path.join(data_folder, "4_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Dictionary to store data
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
    
    # Create the plots
    metrics = ['Efficiency', 'Civility', 'TimeForInterval']
    metric_titles = ['Efficiency', 'Civility', 'Time per Step (seconds)']
    
    for metric, title in zip(metrics, metric_titles):
        # Create figure with 3 subplots stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        for res_idx, resolution in enumerate([100, 200, 300]):
            ax = axes[res_idx]
            
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
                    if metric == 'TimeForInterval':
                        # Divide time values by 100 to get actual time per step
                        mean_values = np.mean([data[metric] / 100.0 for data in group_data], axis=0)
                        std_values = np.std([data[metric] / 100.0 for data in group_data], axis=0)
                    else:
                        # Keep other metrics as they are
                        mean_values = np.mean([data[metric] for data in group_data], axis=0)
                        std_values = np.std([data[metric] for data in group_data], axis=0)
                    
                    steps = np.arange(len(mean_values)) * 100
                    
                    ax.plot(steps, mean_values, 
                           label=f"{'Helbing' if method == 'helbing' else 'Vision'}-based ({agent_count} agents)",
                           marker='o', markersize=4)
                    ax.fill_between(steps, 
                                  mean_values - std_values, 
                                  mean_values + std_values, 
                                  alpha=0.2)
            
            ax.set_title(f'Resolution {resolution}×{resolution}')
            ax.set_xlabel('Simulation Step')
            ax.set_ylabel('Time per Step (seconds)' if metric == 'TimeForInterval' else title)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(f'{title} Over Time', y=0.95, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{metric.lower()}_combined_time_series.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

def write_test_descriptions():
    """
    Test 1: Trail Formation Parameter Analysis
    - Purpose: Evaluate how trail intensity (I) and recovery rate (T) affect trail formation
    - Key Findings:
        * Higher values of both T and I correlate with increased Civility and Efficiency
        * Comfort levels show positive correlation with parameter values
        * Optimal configurations appear around T=15, I=10 or vice versa
        * Visual evidence shows clear trail formation at these values
    - Visualizations:
        * combined_metrics_heatmap_I_and_T
        * final_trails_grid_I_and_T
        * metrics_time_series_I_and_T

    Test 2: Method Parameter Comparison
    - Purpose: Find parameter combinations that produce similar trails between Helbing's and Vision-based methods
    - Key Findings:
        * Best Helbing configuration: Force=2, Sigma=6
        * Closest Vision match: Force=120, Sigma=10
        * Complete parameter matching proved challenging
        * Distinct characteristics remain between methods
    - Visualizations:
        * force_sigma_heatmaps
        * force_sigma_screenshots_comparison

    Test 3: Vision Method Sampling Analysis
    - Purpose: Investigate the impact of sampling point density on trail formation
    - Key Findings:
        * No conclusive evidence for optimal sampling configuration
        * Further investigation needed for minimal sampling requirements
        * Trail quality varies with sampling density
        * More systematic testing required
    - Visualizations:
        * sampling_comparison_heatmaps
        * sampling_comparison_screenshots

    Test 4: Performance Scaling Analysis
    - Purpose: Compare computational efficiency between methods across different scales
    - Key Findings:
        * Vision-based method shows significant performance advantages
        * ~5x speedup observed in high-resolution scenarios
        * Both methods maintain sub-0.02s step time (Unity FixedUpdate requirement)
        * Vision method scales better with increased resolution and agent count
        * Preliminary results promising but require further validation
    - Visualizations:
        * civility_combined_time_series
        * combined_metrics_heatmap
        * efficiency_combined_time_series
        * method_comparison_grid
        * performance_comparison
    """
    pass

def plot_test3_small(data_folder):
    """Plot final screenshots in a grid for the small sampling test"""
    test_folder = os.path.join(data_folder, "3_Sample_points_paired_with_forces_and_sigmas_small")
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Parameters from the test
    forces = [200, 225, 250, 275, 300, 325, 350, 375, 400, 425]
    sigmas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    vision_lengths = [30, 40, 50]
    fovs = [120, 180]
    
    # Create subplots for each vision_length and fov combination
    fig = plt.figure(figsize=(25, 15))
    outer_gs = plt.GridSpec(len(vision_lengths), len(fovs), figure=fig, hspace=0.3, wspace=0.2)
    
    for v_idx, vision_length in enumerate(vision_lengths):
        for f_idx, fov in enumerate(fovs):
            ax = fig.add_subplot(outer_gs[v_idx, f_idx])
            # Move title up by increasing y position
            ax.set_title(f'Vision Length: {vision_length}, FOV: {fov}°', pad=25)
            
            # Create a grid of screenshots for this configuration
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                len(forces), len(sigmas),
                subplot_spec=outer_gs[v_idx, f_idx],
                hspace=0.1, wspace=0.1
            )
            
            # Add sigma values at the top of each column
            for sigma_idx, sigma in enumerate(sigmas):
                ax.text(sigma_idx/len(sigmas) + 1/(2*len(sigmas)), 1.02, 
                       f'σ={sigma}', ha='center', va='bottom', fontsize=8)
            
            # Add force values on the left of each row
            for force_idx, force in enumerate(forces):
                ax.text(-0.02, 1 - (force_idx + 0.5)/len(forces), 
                       f'F={force}', ha='right', va='center', fontsize=8)
            
            for force_idx, force in enumerate(forces):
                for sigma_idx, sigma in enumerate(sigmas):
                    inner_ax = fig.add_subplot(inner_gs[force_idx, sigma_idx])
                    
                    # Find the corresponding simulation folder
                    sim_folder_pattern = f"Vision_arcs=5_first=10_last=20_visionLength={vision_length}_fov={fov}_force={force}_sigma={sigma}"
                    sim_folders = [f for f in os.listdir(test_folder) 
                                 if os.path.isdir(os.path.join(test_folder, f)) 
                                 and sim_folder_pattern in f]
                    
                    if sim_folders:
                        sim_folder = sim_folders[0]
                        screenshot_path = os.path.join(test_folder, sim_folder, "step_1000.png")
                        if os.path.exists(screenshot_path):
                            # Read and crop image
                            img = plt.imread(screenshot_path)
                            h, w = img.shape[:2]
                            crop_left = int(w * 0.2)
                            crop_right = int(w * 0.8)
                            crop_top = int(h * 0.1)
                            crop_bottom = int(h * 0.9)
                            cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]
                            inner_ax.imshow(cropped_img)
                    
                    inner_ax.axis('off')
            
            ax.axis('off')
    
    plt.suptitle('Final States for Different Parameter Combinations\n(5 arcs, 10-20 points)', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'parameter_grid_screenshots.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test5(data_folder):
    """Plot performance comparison with different sampling point configurations"""
    test_folder = os.path.join(data_folder, "5_Sample_points_scaling_test")
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Configurations to analyze
    sampling_configs = [
        {"arcs": 3, "first": 6, "last": 12, "force": 300, "sigma": 10, 
         "label": "Minimal (~30 points, F=300, σ=10)"},
        {"arcs": 5, "first": 10, "last": 20, "force": 180, "sigma": 4, 
         "label": "Medium (~75 points, F=180, σ=4)"},
        {"arcs": 8, "first": 16, "last": 32, "force": 120, "sigma": 4, 
         "label": "High (~192 points, F=120, σ=4)"}
    ]
    
    resolutions = [100, 200, 300, 400, 500]
    agent_counts = [10, 30, 50]
    
    # Create performance comparison plots
    fig, axes = plt.subplots(len(agent_counts), 1, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.4)
    
    for agent_idx, agent_count in enumerate(agent_counts):
        ax = axes[agent_idx]
        
        for config in sampling_configs:
            times = []
            for resolution in resolutions:
                pattern = f"Vision_arcs={config['arcs']}_first={config['first']}_last={config['last']}_force={config['force']}_sigma={config['sigma']}_resolution={resolution}_agents={agent_count}"
                avg_time = 0
                count = 0
                
                # Find matching simulation folders
                for folder in os.listdir(test_folder):
                    if pattern in folder:
                        data_file = os.path.join(test_folder, folder, "data.csv")
                        if os.path.exists(data_file):
                            data = load_data(data_file)
                            if data is not None:
                                data, _ = data
                                avg_time += data['TimeForInterval'].mean()
                                count += 1
                
                if count > 0:
                    times.append(avg_time / count)
                else:
                    times.append(np.nan)
            
            ax.plot(resolutions, times, marker='o', label=config['label'])
        
        ax.set_title(f'Performance Scaling with {agent_count} Agents')
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Average Time per Step (ms)')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle('Performance Comparison with Different Sampling Configurations', y=0.95)
    plt.savefig(os.path.join(results_folder, 'sampling_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test6(data_folder):
    """Plot performance comparison with different sampling point configurations"""
    test_folder = os.path.join(data_folder, "5_Performance_with_varying_agent_count_and_Resolution")
    results_folder = os.path.join(test_folder, "_Results")
    create_screenshot_grid(data_folder)
    create_combined_heatmap(data_folder)
    create_performance_comparison(data_folder)
    create_combined_time_series(data_folder) 

if __name__ == "__main__":
    # Look for data in Unity's default location
    data_folder = "../ExperimentData"
    if not os.path.exists(data_folder):
        print(f"Creating data folder: {os.path.abspath(data_folder)}")
        os.makedirs(data_folder)
    
    #create_screenshot_grid(data_folder)
    #create_combined_heatmap(data_folder)
    #create_performance_comparison(data_folder)
    #create_combined_time_series(data_folder) 
    #plot_test1(data_folder)
    #plot_test2(data_folder)
    #plot_test3(data_folder)
    plot_test4(data_folder)
    plot_test3_small(data_folder)
    plot_test5(data_folder)
    plot_test6(data_folder)