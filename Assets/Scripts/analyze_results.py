import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Fix for colormap
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import itertools
import matplotlib.colors
from PIL import Image  # Change from skimage to PIL

def convert_float(value):
    """Convert comma-formatted number to float"""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def load_data(filename):
    """Load data from CSV file and return DataFrame and config lines"""
    # Normalize path separators and remove .meta extension if present
    filename = os.path.normpath(filename)
    filename = filename.replace('.meta', '')
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
        
    try:
        # First read the configuration section
        config_lines = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i < 20:  # First 20 lines are config
                    config_lines.append(line.strip())
                    
        # Then read the data section
                    data = pd.read_csv(filename, delimiter=':', skiprows=25, 
                                    names=['Step', 'TimeForInterval', 'Efficiency', 'Civility'])
            
            # Convert comma decimals to periods
            for col in ['TimeForInterval', 'Efficiency', 'Civility']:
                data[col] = data[col].apply(lambda x: convert_float(x) if pd.notnull(x) else x)
                    
                if data.empty:
                    print(f"Warning: Empty data in {filename}")
                
            return data, config_lines
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def adjust_plot_height(ax, fig, amount_of_plots):
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
    if legend_height > current_height * amount_of_plots:
        fig.set_size_inches(current_width, legend_height/amount_of_plots)
    
    plt.tight_layout()

def create_comfort_colormap():
    """Create Unity's comfort colormap"""
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
    return LinearSegmentedColormap.from_list('comfort_cmap', list(zip(thresholds, colors)))

def create_test1_time_series(data_folder, test_name, metrics):
    """Create time series plots for Test 1 with all T/I combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Print all folders to see what we have
    print("\nAvailable folders in directory:")
    folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    print(f"Found {len(folders)} total folders (excluding .meta files)")
    
    # Create a dictionary to store all folder paths
    data_paths = {}
    for folder in folders:
        if "Helbing_T=" in folder:
            try:
                # Extract T and I values from folder name
                parts = folder.split('_')
                T = None
                I = None
                for part in parts:
                    if part.startswith('T='):
                        T = int(part.split('=')[1])
                    elif part.startswith('I='):
                        I = int(part.split('=')[1])
                
                # Calculate expected index for this T,I combination
                expected_index = ((T-1) * 20) + I
                
                # Check if this is the folder we want
                if f"_repetition=1_index={expected_index}" in folder:
                    data_paths[(T, I)] = os.path.join(test_folder, folder, "data.csv")
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
                continue
    
    print(f"\nFound {len(data_paths)} valid data paths out of expected 400")
    
    # Create figure with taller plots and bigger text
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9.6*len(metrics)))  # Smaller width since no legend
    if len(metrics) == 1:
        axes = [axes]
    
    # Create distinct colors for each T value
    T_colors = [
        '#FF0000',  # Red
        '#FF4000',  # Orange-Red
        '#FF8000',  # Orange
        '#FFC000',  # Orange-Yellow
        '#FFFF00',  # Yellow
        '#C0FF00',  # Yellow-Green
        '#80FF00',  # Light Green
        '#40FF00',  # Lime Green
        '#00FF00',  # Green
        '#00FF40',  # Green-Cyan
        '#00FF80',  # Light Cyan
        '#00FFC0',  # Cyan
        '#00FFFF',  # Cyan
        '#00C0FF',  # Light Blue
        '#0080FF',  # Blue
        '#0040FF',  # Deep Blue
        '#0000FF',  # Pure Blue
        '#4000FF',  # Blue-Purple
        '#8000FF',  # Purple
        '#C000FF',  # Violet
    ]
    
    # Create distinct colors for I values (will be blended)
    I_colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    # Store line styles
    line_styles = {}
    
    # Pre-calculate all colors and sort by T value
    for T in range(1, 21):
        for I in range(1, 21):
            T_color = np.array(matplotlib.colors.to_rgba(T_colors[T-1]))
            I_color = I_colors[I-1]
            blended_color = (T_color + I_color) / 2
            line_styles[(T,I)] = {'color': blended_color, 'alpha': 0.7}
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_count = 0
        
        # First pass to find the full range of y values
        y_min = float('inf')
        y_max = float('-inf')
        for T in range(20, 0, -1):
            for I in range(1, 21):
                expected_index = ((T-1) * 20) + I
                expected_folder = f"Helbing_T={T}_I={I}_repetition=1_index={expected_index}"
                
                if (T, I) in data_paths and os.path.exists(data_paths[(T, I)]):
                    result = load_data(data_paths[(T, I)])
                    if result:
                        data_df, _ = result
                        if len(data_df) >= 20:
                            y_min = min(y_min, data_df[metric].min())
                            y_max = max(y_max, data_df[metric].max())
        
        # Add 10% padding to y-axis limits
        y_range = y_max - y_min
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Plot lines with known y-axis range
        for T in range(20, 0, -1):
            for I in range(1, 21):
                expected_index = ((T-1) * 20) + I
                expected_folder = f"Helbing_T={T}_I={I}_repetition=1_index={expected_index}"
                
                if (T, I) in data_paths and os.path.exists(data_paths[(T, I)]):
                    result = load_data(data_paths[(T, I)])
                    if result:
                        data_df, _ = result
                        if len(data_df) >= 20:
                            steps = range(100, 2100, 100)
                            ax.plot(steps, data_df[metric], 
                                   **line_styles[(T,I)], linewidth=1.5)  # Thicker lines
                            plotted_count += 1
                        else:
                            print(f"✗ Not enough data points in: {expected_folder}")
                else:
                    print(f"✗ Missing folder: {expected_folder}")
        
        print(f"\nPlotted {plotted_count} lines for {metric}")
        
        ax.set_title(f'{metric} over Time', fontsize=18)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Set the expanded y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # Set x-axis to full range (0 to 2000)
        ax.set_xlim(0, 2000)
    
    # Remove legend - no longer needed
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_folder, '1_metrics_time_series.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_time_series_filtered(data_folder, test_name, metrics):
    """Create time series plots for Test 1 with Efficiency > 60"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Filter data paths to only include those with Efficiency > 60
    data_paths = {}
    # Exclude .meta files
    folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    
    for folder in folders:
        if "Helbing_T=" in folder:
            result = load_data(os.path.join(test_folder, folder, "data.csv"))
            if result:
                data_df, _ = result
                if (data_df['Efficiency'] > 60).any():  # Changed to filter by Efficiency
                    try:
                        parts = folder.split('_')
                        T = next(int(part.split('=')[1]) for part in parts if part.startswith('T='))
                        I = next(int(part.split('=')[1]) for part in parts if part.startswith('I='))
                        expected_index = ((T-1) * 20) + I
                        if f"_repetition=1_index={expected_index}" in folder:
                            data_paths[(T, I)] = os.path.join(test_folder, folder, "data.csv")
                    except Exception as e:
                        continue

    # Create figure and axes with bigger text
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9.6*len(metrics)))  # Smaller width since no legend
    if len(metrics) == 1:
        axes = [axes]
    
    # Create distinct colors for T values
    T_colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    I_colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    line_styles = {}
    
    for T in range(1, 21):
        for I in range(1, 21):
            T_color = T_colors[T-1]
            I_color = I_colors[I-1]
            blended_color = (np.array(T_color) + np.array(I_color)) / 2
            line_styles[(T,I)] = {'color': blended_color, 'alpha': 0.7}

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_count = 0
        
        for (T, I) in data_paths:
            result = load_data(data_paths[(T, I)])
            if result:
                data_df, _ = result
                if len(data_df) >= 20:
                    steps = range(100, 2100, 100)
                    ax.plot(steps, data_df[metric], **line_styles[(T,I)], linewidth=1.5)  # Thicker lines
                    plotted_count += 1
        
        print(f"\nPlotted {plotted_count} lines for {metric}")
        
        ax.set_title(f'{metric} over Time (Efficiency > 60)', fontsize=18)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, 2000)

    # Remove legend - no longer needed
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '1_metrics_time_series_with_Efficiency_over_60.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_specific_screenshot_grid(data_folder, test_name):
    """Create a focused screenshot grid with only specific T,I combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Specific combinations to show
    combinations = [
        {'T': 9, 'I': 6, 'label': 'Well spread out agents\n(T=9, I=6)'},
        {'T': 9, 'I': 7, 'label': 'Small change, big difference\n(T=9, I=7)'},
        {'T': 11, 'I': 16, 'label': 'Extreme case\n(T=11, I=16)'},
        {'T': 18, 'I': 7, 'label': 'Another extreme case\n(T=18, I=7)'}
    ]
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, combo in enumerate(combinations):
        ax = axes[idx]
        T, I = combo['T'], combo['I']
        
        # Find matching folder
        expected_index = ((T-1) * 20) + I
        pattern = f"Helbing_T={T}_I={I}_repetition=1_index={expected_index}"
        
        matching_folders = [f for f in os.listdir(test_folder) 
                          if pattern in f and not f.endswith('.meta')]
        
        if matching_folders:
            folder = matching_folders[0]
            screenshot_path = os.path.join(test_folder, folder, "step_2000.png")
            
            if os.path.exists(screenshot_path):
                img = plt.imread(screenshot_path)
                # Crop the image
                h, w = img.shape[:2]
                left = int(w * 0.25)
                right = int(w * 0.75)
                top = int(h * 0.10)
                bottom = int(h * 0.90)
                cropped_img = img[top:bottom, left:right]
                ax.imshow(cropped_img)
            else:
                ax.text(0.5, 0.5, "No screenshot found", ha='center', va='center', fontsize=14)
        else:
            ax.text(0.5, 0.5, "No matching folder found", ha='center', va='center', fontsize=14)
        
        ax.set_title(combo['label'], fontsize=16, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '1_screenshot_grid_mini.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_comprehensive_screenshot_grid(data_folder, test_name):
    """Create comprehensive screenshot grid for Test 1 with all T,I combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create figure for 20x20 grid
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(20, 20, wspace=0.05, hspace=0.05)
    
    # Get all folders
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    
    # Create a dictionary to store folder paths
    data_paths = {}
    for folder in all_folders:
        if "Helbing_T=" in folder:
            try:
                # Extract T and I values from folder name
                parts = folder.split('_')
                T = None
                I = None
                for part in parts:
                    if part.startswith('T='):
                        T = int(part.split('=')[1])
                    elif part.startswith('I='):
                        I = int(part.split('=')[1])
                
                # Calculate expected index for this T,I combination
                expected_index = ((T-1) * 20) + I
                
                # Check if this is the folder we want (repetition=1)
                if f"_repetition=1_index={expected_index}" in folder:
                    data_paths[(T, I)] = folder
            except Exception as e:
                continue
    
    print(f"Found {len(data_paths)} valid T,I combinations for comprehensive grid")
    
    # Create grid of screenshots
    for T in range(1, 21):
        for I in range(1, 21):
            # Position in grid (T=1 at top, T=20 at bottom)
            row = T - 1
            col = I - 1
            
            ax = plt.Subplot(fig, gs[row, col])
            fig.add_subplot(ax)
            
            if (T, I) in data_paths:
                folder = data_paths[(T, I)]
                screenshot_path = os.path.join(test_folder, folder, "step_2000.png")
                
                if os.path.exists(screenshot_path):
                    img = plt.imread(screenshot_path)
                    # Crop the image
                    h, w = img.shape[:2]
                    left = int(w * 0.25)
                    right = int(w * 0.75)
                    top = int(h * 0.10)
                    bottom = int(h * 0.90)
                    cropped_img = img[top:bottom, left:right]
                    ax.imshow(cropped_img)
                else:
                    ax.text(0.5, 0.5, "No img", ha='center', va='center', fontsize=6)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=6)
            
            # Add labels only on edges
            if T == 1:  # Top row
                ax.set_title(f'I={I}', fontsize=8)
            if I == 1:  # Left column
                ax.set_ylabel(f'T={T}', fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Trail Formation with Different T and I Values (Complete Grid)', fontsize=20, y=0.98)
    
    plt.savefig(os.path.join(results_folder, '1_screenshot_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_heatmap(data_folder, test_name, metrics):
    """Create vertically stacked heatmaps for Test 1"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Create figure with vertical stack (2 rows, 1 column) and much larger size
    fig, axes = plt.subplots(2, 1, figsize=(20, 28))  # Vertical stack, much larger
    
    # Create and register the comfort colormap
    comfort_cmap = create_comfort_colormap()
    
    for idx, metric in enumerate(metrics):
        data = np.zeros((20, 20))  # T x I grid
        
        # Collect average values for each T,I combination
        for T in range(1, 21):
            for I in range(1, 21):
                pattern = f"Helbing_T={T}_I={I}"
                values = []
                
                for folder in os.listdir(test_folder):
                    if pattern in folder:
                        data_file = os.path.join(test_folder, folder, "data.csv")
                        if os.path.exists(data_file):
                            result = load_data(data_file)
                            if result:
                                data_df, _ = result
                                # Take average of all data points
                                values.append(data_df[metric].mean())
                
                if values:
                    data[20-T, I-1] = np.mean(values)  # Average across repetitions
        
        # Plot heatmap with much bigger text and numbers
        hm = sns.heatmap(data, 
                   ax=axes[idx],
                   cmap=comfort_cmap if metric == 'Civility' else 'viridis',
                   xticklabels=range(1, 21),
                   yticklabels=range(20, 0, -1),  # Reverse T labels
                   annot=True,  # Add numbers inside boxes
                   fmt='.1f',  # Format numbers to 1 decimal place
                   annot_kws={'size': 20, 'weight': 'bold'})  # Much bigger annotation text with bold
        
        # Increase colorbar/legend font size
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)  # Much bigger colorbar labels
        
        axes[idx].set_xlabel('I Value (Footstep Intensity)', fontsize=28)  # Even bigger with descriptive label
        axes[idx].set_ylabel('T Value (Trail Recovery Rate)', fontsize=28)  # Even bigger with descriptive label
        axes[idx].set_title(f'{metric}', fontsize=32, pad=20)  # Even bigger title with more padding
        
        # Make tick labels much bigger
        axes[idx].tick_params(axis='both', which='major', labelsize=24)
        
        # Add a subtle grid to help reading
        axes[idx].grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for suptitle
    plt.savefig(os.path.join(results_folder, '1_combined_metrics_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_heatmaps(data_folder, test_name, metrics, methods):
    """Create 2x2 grid of heatmaps for Test 2"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create and register the comfort colormap
    comfort_cmap = create_comfort_colormap()
    
    # Define expected parameter ranges
    param_ranges = {
        'Helbing': {
            'forces': [0.5, 1, 1.5, 1.75, 2, 2.5, 2.75, 3, 4],
            'sigmas': list(range(4, 13))  # 4 to 12
        },
        'Vision': {
            'forces': [175, 200, 225, 375, 400, 425, 675, 700, 725],
            'sigmas': [4, 5, 6, 11, 12, 13, 19, 20, 21]
        }
    }
    
    # Create figure with 2x2 grid - dramatically larger
    fig, axes = plt.subplots(2, 2, figsize=(36, 32))
    
    # Process each method (columns)
    for method_idx, method in enumerate(methods):
        forces = param_ranges[method]['forces']
        sigmas = param_ranges[method]['sigmas']
        
        # Process each metric (rows)
        for metric_idx, metric in enumerate(metrics):
            # Create data matrix
            data = np.zeros((len(forces), len(sigmas)))
            data[:] = np.nan  # Fill with NaN initially
            
            # Fill matrix with metric values
            for i, force in enumerate(forces):
                for j, sigma in enumerate(sigmas):
                    # Handle comma vs period in force values
                    force_str = str(force).replace('.', ',')
                    
                    # Find matching folder by index range
                    index_range = range(1, 82) if method == 'Helbing' else range(82, 163)
                    values = []
                    
                    for index in index_range:
                        folder = f"{method}_force={force_str}_sigma={sigma}_index={index}"
                        data_path = os.path.join(test_folder, folder, 'data.csv')
                        if os.path.exists(data_path):
                            df, _ = load_data(data_path)
                            if df is not None:
                                values.append(df[metric].mean())
                    
                    if values:
                        data[i, j] = np.mean(values)
            
            ax = axes[metric_idx, method_idx]
            
            # Set vmin/vmax for Civility to 0-20
            kwargs = {}
            if metric == 'Civility':
                kwargs = {'vmin': 0, 'vmax': 20}
            
            # Plot heatmap with dramatically bigger text and numbers
            hm = sns.heatmap(data, 
                       ax=ax,
                       cmap=comfort_cmap if metric == 'Civility' else 'viridis',
                       xticklabels=[f'{s:.1f}' for s in sigmas],
                       yticklabels=[f'{f:.1f}' for f in forces],
                       annot=True,
                       fmt='.1f',
                       annot_kws={'size': 24, 'weight': 'bold'},  # Dramatically bigger annotation text with bold
                       **kwargs)
            
            # Increase colorbar/legend font size
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=22)  # Much bigger colorbar labels
            
            ax.set_xlabel('Sigma (σ) - Vision Range Parameter', fontsize=32)  # Much bigger with descriptive label
            ax.set_ylabel('Force - Path Following Strength', fontsize=32)  # Much bigger with descriptive label
            ax.set_title(f'{method} {metric}', fontsize=36, pad=20)  # Much bigger title with padding
            ax.tick_params(axis='both', which='major', labelsize=28)  # Much bigger tick labels
            
            # Add a subtle grid to help reading
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Invert y-axis so lowest force is at bottom
            ax.invert_yaxis()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for suptitle
    plt.savefig(os.path.join(results_folder, '2_parameter_heatmaps.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_specific_screenshot_grid(data_folder, test_name):
    """Create screenshot grid with only specific Helbing combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Specific Helbing combinations to show
    combinations = [
        {'force': 2, 'sigma': 7, 'label': 'Good trails\n(F=2, σ=7)'},
        {'force': 0.5, 'sigma': 7, 'label': 'Too little force\n(F=0.5, σ=7)'},
        {'force': 2, 'sigma': 4, 'label': 'Too little sigma\n(F=2, σ=4)'},
        {'force': 2.5, 'sigma': 4, 'label': 'Pooling behavior\n(F=2.5, σ=4)'},
        {'force': 3, 'sigma': 10, 'label': 'Pooling behavior\n(F=3, σ=10)'}
    ]
    
    # Create single row layout with much larger figure
    fig, axes = plt.subplots(1, 5, figsize=(35, 8))  # Much larger figure
    
    for idx, combo in enumerate(combinations):
        ax = axes[idx]
        force = combo['force']
        sigma = combo['sigma']
        
        # Handle comma vs period in force values
        force_str = str(force).replace('.', ',')
        folder_pattern = f"Helbing_force={force_str}_sigma={sigma}"
        
        # Find matching folder
        matching_folders = [f for f in os.listdir(test_folder)
                          if folder_pattern in f and not f.endswith('.meta')]
        
        if matching_folders:
            folder = matching_folders[0]  # Take first matching folder
            screenshot_path = os.path.join(test_folder, folder, "step_2000.png")
            
            if os.path.exists(screenshot_path):
                img = plt.imread(screenshot_path)
                h, w = img.shape[:2]
                left = int(w * 0.25)
                right = int(w * 0.75)
                top = int(h * 0.10)
                bottom = int(h * 0.90)
                cropped_img = img[top:bottom, left:right]
                ax.imshow(cropped_img)
            else:
                ax.text(0.5, 0.5, "No screenshot found", ha='center', va='center', fontsize=20)  # Bigger error text
        else:
            ax.text(0.5, 0.5, "No matching folder found", ha='center', va='center', fontsize=20)  # Bigger error text
        
        ax.set_title(combo['label'], fontsize=24, pad=15, weight='bold')  # Much bigger titles with bold
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for suptitle
    plt.savefig(os.path.join(results_folder, '2_parameter_grid_small.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_comprehensive_screenshot_grid(data_folder, test_name):
    """Create comprehensive screenshot grid for Test 2 with all parameter combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Define expected parameter ranges
    param_ranges = {
        'Helbing': {
            'forces': [0.5, 1, 1.5, 1.75, 2, 2.5, 2.75, 3, 4],
            'sigmas': list(range(4, 13))  # 4 to 12
        },
        'Vision': {
            'forces': [175, 200, 225, 375, 400, 425, 675, 700, 725],
            'sigmas': [4, 5, 6, 11, 12, 13, 19, 20, 21]
        }
    }
    
    # Create subplot for each method
    fig = plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    for method_idx, method in enumerate(['Helbing', 'Vision']):
        forces = param_ranges[method]['forces']
        sigmas = param_ranges[method]['sigmas']
        
        n_force_groups = len(forces) // 3
        n_sigma_groups = len(sigmas) // 3
        
        group_gs = gridspec.GridSpecFromSubplotSpec(n_force_groups, n_sigma_groups,
                                                  subplot_spec=gs[method_idx],
                                                  wspace=0, hspace=0.05)
        
        # Create sub-grids for each group
        inner_gs = []
        for i in range(n_force_groups):
            for j in range(n_sigma_groups):
                sub_gs = gridspec.GridSpecFromSubplotSpec(3, 3,
                                                        subplot_spec=group_gs[i, j],
                                                        wspace=0, hspace=0.1)   
                for si, sj in itertools.product(range(3), range(3)):
                    inner_gs.append((sub_gs[si, sj], i*3 + si, j*3 + sj))
        
        plt.figtext(0.02, 0.75 - method_idx * 0.5, method, 
                   rotation=90, va='center', fontsize=12)
        
        # Store axes that need boxes - store the group_gs instead of individual axes
        box_groups = []
        if method == 'Vision':
            box_groups = [
                {'pos': (0, 0), 'subplot_spec': group_gs[0, 0]},  # Top left
                {'pos': (1, 1), 'subplot_spec': group_gs[1, 1]},  # Middle
                {'pos': (2, 2), 'subplot_spec': group_gs[2, 2]}   # Bottom right
            ]
        
        # Create grid of screenshots
        for i, force in enumerate(forces):
            for j, sigma in enumerate(sigmas):
                # Calculate the index in inner_gs
                group_i, group_j = i // 3, j // 3
                sub_i, sub_j = i % 3, j % 3
                idx = group_i * (n_sigma_groups * 9) + group_j * 9 + sub_i * 3 + sub_j
                ax = plt.Subplot(fig, inner_gs[idx][0])
                
                fig.add_subplot(ax)
                
                # Handle comma vs period in force values
                force_str = str(force).replace('.', ',')
                folder_pattern = f"{method}_force={force_str}_sigma={sigma}"
                
                # Find matching folder
                matching_folders = [f for f in os.listdir(test_folder)
                                 if folder_pattern in f and not f.endswith('.meta')]
                
                screenshot_path = None
                if matching_folders:
                    folder = matching_folders[0]  # Take first matching folder
                    screenshot_path = os.path.join(test_folder, folder, "step_2000.png")
                
                if screenshot_path and os.path.exists(screenshot_path):
                    img = plt.imread(screenshot_path)
                    h, w = img.shape[:2]
                    left = int(w * 0.25)
                    right = int(w * 0.75)
                    top = int(h * 0.10)
                    bottom = int(h * 0.90)
                    cropped_img = img[top:bottom, left:right]
                    
                    # Resize using PIL with 50% larger dimensions
                    h_crop = bottom - top
                    w_crop = right - left
                    new_h = int(h_crop * 1.5)
                    new_w = int(w_crop * 1.5)
                    
                    cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
                    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    resized_img = np.array(resized_img) / 255.0
                    
                    ax.imshow(resized_img)
                else:
                    ax.text(0.5, 0.5, "No image", ha='center', va='center')
                
                # Add labels
                if i == 0:  # Top row
                    ax.set_title(f'σ={sigma:.1f}')
                if j == 0:  # Left column
                    ax.set_ylabel(f'f={force:.1f}')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add black boxes around specified groups in Vision method
        if method == 'Vision':
            for idx, group in enumerate(box_groups):
                # Get the bounding box of the entire group
                bbox = group['subplot_spec'].get_position(fig)
                
                # Select color based on idx
                color = {
                    0: 'gold',
                    1: 'cyan',
                    2: 'darkviolet'
                }.get(idx, 'black')
                
                # Create a rectangle patch
                rect = plt.Rectangle(
                    (bbox.x0 -0.015+0.025*idx, bbox.y0 -0.035 -0.013*idx),
                    bbox.width + 0.005,
                    bbox.height + 0.012,
                    fill=False,
                    color=color,
                    linewidth=2,
                    transform=fig.transFigure
                )
                fig.add_artist(rect)
    
    plt.figtext(0.05, 0.5, "Force", rotation=90, va='center', fontsize=12)
    plt.suptitle("Trail Formation with Different Force and Sigma Values", y=0.95)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(os.path.join(results_folder, '2_parameter_grid.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_combined_time_series(data_folder, test_name, metrics, methods):
    """Create time series plots comparing both methods without legend"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9.6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    # Define parameter ranges for color interpolation
    param_ranges = {
        'Helbing': {
            'forces': [0.5, 1, 1.5, 1.75, 2, 2.5, 2.75, 3, 4],
            'sigmas': list(range(4, 13))  # 4 to 12
        },
        'Vision': {
            'forces': [175, 200, 225, 375, 400, 425, 675, 700, 725],
            'sigmas': [4, 5, 6, 11, 12, 13, 19, 20, 21]
        }
    }
    
    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Plot data for each method
        for method in methods:
            # Get all matching folders for this method
            folders = [f for f in os.listdir(test_folder) 
                      if f.startswith(method) and not f.endswith('.meta')]
            
            # Get parameter ranges for this method
            force_range = param_ranges[method]['forces']
            sigma_range = param_ranges[method]['sigmas']
            
            # Plot each configuration
            for folder in folders:
                result = load_data(os.path.join(test_folder, folder, "data.csv"))
                if result:
                    data_df, _ = result
                    if len(data_df) >= 20:
                        # Extract force and sigma from folder name
                        parts = folder.split('_')
                        try:
                            force = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('force='))
                            sigma = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('sigma='))
                            
                            # Calculate force color (interpolate between blue for Helbing, red for Vision)
                            force_min = min(force_range)
                            force_max = max(force_range)
                            force_t = (force - force_min) / (force_max - force_min)
                            
                            if method == 'Helbing':
                                force_color = np.array([0, 0, force_t])  # Blue interpolation
                            else:
                                force_color = np.array([force_t, 0, 0])  # Red interpolation
                            
                            # Calculate sigma color (interpolate on green for both)
                            sigma_min = min(sigma_range)
                            sigma_max = max(sigma_range)
                            sigma_t = (sigma - sigma_min) / (sigma_max - sigma_min)
                            sigma_color = np.array([0, sigma_t, 0])
                            
                            # Combine colors
                            combined_color = force_color + sigma_color
                            # Normalize if any component > 1
                            max_component = max(combined_color)
                            if max_component > 1:
                                combined_color = combined_color / max_component
                            
                            ax.plot(data_df['Step'], data_df[metric],
                                   color=combined_color, alpha=0.3, linewidth=1.5)
                                     
                        except Exception as e:
                            print(f"Error processing folder {folder}: {e}")
                            continue
        
        ax.set_title(f'{metric} over Time', fontsize=18)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
    
    # Remove legend - no longer needed
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '2_combined_time_series.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_similar_combinations_time_series(data_folder, test_name):
    """Create time series plots for the same combinations as the screenshots"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Define the combinations we want to show (same as screenshots)
    combinations = [
        {'method': 'Helbing', 'force': 2, 'sigma': 7},
        {'method': 'Vision', 'force': 200, 'sigma': 5},
        {'method': 'Vision', 'force': 425, 'sigma': 12},
        {'method': 'Vision', 'force': 700, 'sigma': 20}
    ]
    
    # Create figure with subplots for each metric
    metrics = ['Efficiency', 'Civility']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 12))
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Plot each combination
        for combo in combinations:
            # Find matching folder
            force_str = str(combo['force']).replace('.', ',')
            folder_pattern = f"{combo['method']}_force={force_str}_sigma={combo['sigma']}"
            
            matching_folders = [f for f in os.listdir(test_folder)
                              if folder_pattern in f and not f.endswith('.meta')]
            
            if matching_folders:
                folder = matching_folders[0]
                data_path = os.path.join(test_folder, folder, 'data.csv')
                
                if os.path.exists(data_path):
                    data_df, _ = load_data(data_path)
                    if data_df is not None:
                        # Plot the line
                        ax.plot(data_df['Step'], data_df[metric],
                               label=f"{combo['method']} (f={combo['force']}, σ={combo['sigma']})",
                               linewidth=2)
        
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.set_title(f'{metric} over Time', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '2_similar_combinations_time_series.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_similar_combinations_screenshots(data_folder, test_name):
    """Create grid of screenshots showing time evolution for specific parameter combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    combinations = [
        {'method': 'Helbing', 'force': 2, 'sigma': 7},
        {'method': 'Vision', 'force': 200, 'sigma': 5},
        {'method': 'Vision', 'force': 425, 'sigma': 12},
        {'method': 'Vision', 'force': 700, 'sigma': 20}
    ]
    
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(4, 10, figure=fig, hspace=0.1)
    
    for row, combo in enumerate(combinations):
        force_str = str(combo['force']).replace('.', ',')
        folder_pattern = f"{combo['method']}_force={force_str}_sigma={combo['sigma']}"
        
        matching_folders = [f for f in os.listdir(test_folder)
                          if folder_pattern in f and not f.endswith('.meta')]
        
        if matching_folders:
            folder = matching_folders[0]
            
            for col, step in enumerate(range(1, 20, 2)):  # 10 columns, steps 1,3,5,...,19
                ax = plt.Subplot(fig, gs[row, col])
                fig.add_subplot(ax)
                
                screenshot_path = os.path.join(test_folder, folder, f"step_{step*100}.png")
                
                if os.path.exists(screenshot_path):
                    img = plt.imread(screenshot_path)
                    h, w = img.shape[:2]
                    left = int(w * 0.25)
                    right = int(w * 0.75)
                    top = int(h * 0.10)
                    bottom = int(h * 0.90)
                    cropped_img = img[top:bottom, left:right]
                    
                    # Resize using PIL with 50% larger dimensions
                    h_crop = bottom - top
                    w_crop = right - left
                    new_h = int(h_crop * 1.5)
                    new_w = int(w_crop * 1.5)
                    
                    cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
                    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    resized_img = np.array(resized_img) / 255.0
                    
                    ax.imshow(resized_img)
                else:
                    ax.text(0.5, 0.5, "No image", ha='center', va='center')
                
                if col == 0:
                    label = f"{combo['method']}\nf={combo['force']}, σ={combo['sigma']}"
                    ax.set_ylabel(label, fontsize=14)
                if row == 0:
                    ax.set_title(f"t={step*100}", fontsize=14)
                
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.savefig(os.path.join(results_folder, '2_similar_combinations_screenshots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test1(data_folder):
    """Plot all visualizations for Test 1 in parallel"""
    test_name = "1_I_and_T_values_for_different_trails"
    metrics = ['Efficiency', 'Civility']
    
    print("Starting 1_combined_metrics_heatmap...")
    create_test1_heatmap(data_folder, test_name, metrics)
    print("Completed 1_combined_metrics_heatmap")
    
    print("Starting 1_metrics_time_series...")
    create_test1_time_series(data_folder, test_name, metrics)
    print("Completed 1_metrics_time_series")
    
    print("Starting 1_metrics_time_series_with_Efficiency_over_60...")
    create_test1_time_series_filtered(data_folder, test_name, metrics)
    print("Completed 1_metrics_time_series_with_Efficiency_over_60")
    
    print("Starting 1_screenshot_grid (comprehensive)...")
    create_test1_comprehensive_screenshot_grid(data_folder, test_name)
    print("Completed 1_screenshot_grid (comprehensive)")
    
    print("Starting 1_screenshot_grid_mini...")
    create_test1_specific_screenshot_grid(data_folder, test_name)
    print("Completed 1_screenshot_grid_mini")

def plot_test2(data_folder):
    """Plot results for Method Parameter Comparison"""
    test_name = "2_Force_and_Sigma_Values_to_match_paths"
    
    print("Starting 2_combined_time_series...")
    create_test2_combined_time_series(data_folder, test_name,
                              metrics=['Efficiency', 'Civility'],
                              methods=['Helbing', 'Vision'])
    print("Completed 2_combined_time_series")
    
    print("Starting 2_parameter_heatmap...")
    create_test2_heatmaps(data_folder, test_name,
                           metrics=['Efficiency', 'Civility'],
                           methods=['Helbing', 'Vision'])
    print("Completed 2_parameter_heatmap")
    
    print("Starting 2_parameter_grid...")
    create_test2_comprehensive_screenshot_grid(data_folder, test_name)
    print("Completed 2_parameter_grid")
    
    print("Starting 2_parameter_grid_small...")
    create_test2_specific_screenshot_grid(data_folder, test_name)
    print("Completed 2_parameter_grid_small")
    
    print("Starting 2_similar_combinations_time_series...")
    create_test2_similar_combinations_time_series(data_folder, test_name)
    print("Completed 2_similar_combinations_time_series")
    
    print("Starting 2_similar_combinations_screenshots...")
    create_test2_similar_combinations_screenshots(data_folder, test_name)
    print("Completed 2_similar_combinations_screenshots")

def plot_test4(data_folder):
    """Plot results for Sample Points Scaling Test"""
    test_name = "5_Sample_points_scaling_test"
    
    methods = [
        'Helbing',
        'Vision_arcs=3_first=5_last=5_force=1000_sigma=12',
        'Vision_arcs=5_first=10_last=20_force=425_sigma=12',
        'Vision_arcs=10_first=20_last=20_force=100_sigma=12',
        'Vision_arcs=20_first=50_last=50_force=20_sigma=12'
    ]
    method_labels = [
        'Helbing',
        'Vision (30 pts)',
        'Vision (75 pts)',
        'Vision (200 pts)',
        'Vision (1000 pts)'
    ]
    
    print("Starting 4_performance_bars...")
    create_performance_bar_graph(data_folder, test_name,
                               methods=methods,
                               method_labels=method_labels,
                               filename='4_performance_bars.png')
    print("Completed 4_performance_bars")
    
    print("Starting 4_performance_by_resolution...")
    create_performance_by_resolution(data_folder, test_name,
                                   methods=methods,
                                   method_labels=method_labels)
    print("Completed 4_performance_by_resolution")
    
    print("Starting 4_performance_bars_700...")
    create_performance_bar_graph_700(data_folder, test_name,
                                   methods=methods,
                                   method_labels=method_labels)
    print("Completed 4_performance_bars_700")
    
    print("Starting 4_time_series_by_resolution...")
    create_test4_time_series_by_resolution(data_folder, test_name,
                                         methods=methods,
                                         method_labels=method_labels)
    print("Completed 4_time_series_by_resolution")
    
    print("Starting 4_screenshot_grid...")
    create_test4_screenshot_grid(data_folder, test_name,
                                methods=methods,
                                method_labels=method_labels)
    print("Completed 4_screenshot_grid")
    
    print("Starting 4_screenshot_grid_full...")
    create_test4_full_screenshot_grid(data_folder, test_name,
                                    methods=methods,
                                    method_labels=method_labels)
    print("Completed 4_screenshot_grid_full")

def create_resolution_agent_heatmap(data_folder, test_name, metrics, method, filename):
    """Create heatmap of metrics vs resolution and agent count"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Get all folders for this method
    folders = [f for f in os.listdir(test_folder) 
              if method in f and not f.endswith('.meta')]
    
    # Extract unique resolution and agent values
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in folders if 'resolution=' in f))
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in folders if 'agents=' in f))
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for metric_idx, metric in enumerate(metrics):
        # Create data matrix
        data = np.zeros((len(agent_counts), len(resolutions)))
        
        for i, agents in enumerate(agent_counts):
            for j, res in enumerate(resolutions):
                # Get all matching folders (all repetitions)
                matching_folders = [f for f in folders 
                                  if f'resolution={res}' in f 
                                  and f'agents={agents}' in f]
                
                # Calculate average metric value across repetitions
                values = []
                for folder in matching_folders:
                    data_path = os.path.join(test_folder, folder, 'data.csv')
                    if os.path.exists(data_path):
                        df, _ = load_data(data_path)
                        if df is not None:
                            values.append(df[metric].mean())
                
                if values:
                    data[i, j] = np.mean(values)
        
        # Create heatmap
        ax = axes[metric_idx]
        sns.heatmap(data, ax=ax, 
                   xticklabels=resolutions,
                   yticklabels=agent_counts,
                   cmap='viridis')
        
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Agent Count')
        ax.set_title(f'{metric} - {method}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_time_series_with_performance(data_folder, test_name, methods, method_labels, filename):
    """Create combined time series plot with three subplots including performance"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    metrics = ['Efficiency', 'Civility', 'TimeForInterval']
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for method, label in zip(methods, method_labels):
            # Get all folders for this method
            folders = [f for f in os.listdir(test_folder) 
                      if method in f and not f.endswith('.meta')]
            
            # Calculate average across all configurations and repetitions
            dfs = []
            for folder in folders:
                data_path = os.path.join(test_folder, folder, 'data.csv')
                if os.path.exists(data_path):
                    df, _ = load_data(data_path)
                    if df is not None:
                        dfs.append(df[['Step', metric]])
            
            if dfs:
                # Calculate mean and std
                mean_df = pd.concat(dfs).groupby('Step').mean()
                std_df = pd.concat(dfs).groupby('Step').std()
                
                # Plot mean line with confidence interval
                ax.plot(mean_df.index, mean_df[metric], label=label)
                ax.fill_between(mean_df.index,
                              mean_df[metric] - std_df[metric],
                              mean_df[metric] + std_df[metric],
                              alpha=0.2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over Time')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_bar_graph(data_folder, test_name, methods, method_labels, filename):
    """Create single bar graph showing all performance data with error bars"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in all_folders if 'agents=' in f))
    
    combinations = [(res, agents) for res in resolutions for agents in agent_counts]
    
    fig, ax = plt.subplots(figsize=(24, 10))  # Larger figure
    
    bar_width = 0.15
    x = np.arange(len(combinations))
    
    # Use distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        mins = []
        maxs = []
        for res, agents in combinations:
            folders = [f for f in os.listdir(test_folder) 
                      if method in f and f'resolution={res}' in f and f'agents={agents}' in f]
            run_means = []
            for folder in folders:
                data_path = os.path.join(test_folder, folder, 'data.csv')
                if os.path.exists(data_path):
                    df, _ = load_data(data_path)
                    if df is not None:
                        run_means.append(df['TimeForInterval'].mean() * 10)  # Multiply by 10
            
            if run_means:
                means.append(np.mean(run_means))
                mins.append(np.min(run_means))
                maxs.append(np.max(run_means))
            else:
                means.append(0)
                mins.append(0)
                maxs.append(0)
        
        yerr = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]
        ax.bar(x + i*bar_width, means, bar_width, label=label, color=color)
        ax.errorbar(x + i*bar_width, means, yerr=yerr, fmt='none', color='black', capsize=5)
    
    ax.set_xlabel('Resolution-Agent Count Combinations', fontsize=20)  # Bigger
    ax.set_ylabel('Average Time per Step (ms)', fontsize=20)  # Bigger
    ax.set_title('Performance by Resolution and Agent Count', fontsize=24)  # Bigger
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels([f'R{res}-A{agents}' for res, agents in combinations], rotation=45, fontsize=16)  # Bigger
    ax.tick_params(axis='both', which='major', labelsize=18)  # Bigger
    ax.yaxis.set_major_locator(MultipleLocator(50))  # Reduce y-axis steps
    ax.legend(fontsize=16)  # Bigger
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_bar_graph_700(data_folder, test_name, methods, method_labels):
    """Create bar graph for resolution 700 only"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in all_folders if 'agents=' in f))
    
    fig, ax = plt.subplots(figsize=(18, 10))  # Larger figure
    
    bar_width = 0.15
    x = np.arange(len(agent_counts))
    
    # Use distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        mins = []
        maxs = []
        for agents in agent_counts:
            folders = [f for f in os.listdir(test_folder) 
                      if method in f and 'resolution=700' in f and f'agents={agents}' in f]
            run_means = []
            for folder in folders:
                data_path = os.path.join(test_folder, folder, 'data.csv')
                if os.path.exists(data_path):
                    df, _ = load_data(data_path)
                    if df is not None:
                        run_means.append(df['TimeForInterval'].mean() * 10)  # Multiply by 10
            
            if run_means:
                means.append(np.mean(run_means))
                mins.append(np.min(run_means))
                maxs.append(np.max(run_means))
            else:
                means.append(0)
                mins.append(0)
                maxs.append(0)
        
        yerr = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]
        ax.bar(x + i*bar_width, means, bar_width, label=label, color=color)
        ax.errorbar(x + i*bar_width, means, yerr=yerr, fmt='none', color='black', capsize=5)
    
    ax.set_xlabel('Agent Count', fontsize=20)  # Bigger
    ax.set_ylabel('Average Time per Step (ms)', fontsize=20)  # Bigger
    ax.set_title('Performance at Resolution 700', fontsize=24)  # Bigger
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(agent_counts, fontsize=18)  # Bigger
    ax.tick_params(axis='both', which='major', labelsize=18)  # Bigger
    ax.yaxis.set_major_locator(MultipleLocator(50))  # Reduce y-axis steps
    ax.legend(fontsize=16)  # Bigger
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '4_performance_bars_700.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test4_time_series_by_resolution(data_folder, test_name, methods, method_labels):
    """Create separate time series plots for each resolution"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in all_folders if 'agents=' in f))
    
    # Create different line styles and distinct colors
    linestyles = ['-', '--', ':', '-.', '-']
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # Create a separate plot for each resolution
    for resolution in resolutions:
        # Only create plot for resolution 700
        if resolution != 700:
            continue
            
        fig, ax = plt.subplots(figsize=(18, 10))  # Larger figure
        
        # Track if we've added a method to the legend
        legend_added = {method: False for method in methods}
        
        for method_idx, (method, method_label) in enumerate(zip(methods, method_labels)):
            for agents in agent_counts:
                pattern = f"{method}_resolution={resolution}_agents={agents}"
                matching_folders = [f for f in os.listdir(test_folder) 
                                  if pattern in f and not f.endswith('.meta')]
                
                times = []
                for folder in matching_folders:
                    data_path = os.path.join(test_folder, folder, 'data.csv')
                    if os.path.exists(data_path):
                        df, _ = load_data(data_path)
                        if df is not None:
                            times.append(df['TimeForInterval'].values)
                
                if times:
                    avg_times = np.mean(times, axis=0) * 10  # Multiply by 10 to get ms per step
                    steps = np.arange(100, (len(avg_times) + 1) * 100, 100)
                    
                    # Only add to legend if this method hasn't been added yet
                    label = method_label if not legend_added[method] else None
                    if label:
                        legend_added[method] = True
                        
                    ax.plot(steps, avg_times, 
                           label=label,
                           linestyle=linestyles[method_idx],
                           color=colors[method_idx],
                           alpha=0.7,
                           linewidth=2)  # Thicker lines
        
        ax.set_xlabel('Step', fontsize=20)  # Bigger
        ax.set_ylabel('Time per Step (ms)', fontsize=20)  # Bigger
        ax.set_title(f'Computation Time per Step (Resolution {resolution})', fontsize=24)  # Bigger
        ax.legend(loc='upper left', fontsize=16)  # Bigger legend
        ax.tick_params(axis='both', which='major', labelsize=18)  # Bigger
        ax.grid(True)
        ax.yaxis.set_major_locator(MultipleLocator(100))  # Set y-axis ticks every 100ms
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'4_time_series_res_{resolution}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_test4_screenshot_grid(data_folder, test_name, methods, method_labels):
    """Create grid of final screenshots for each configuration"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create a 1x5 grid (methods side by side) - much larger figure
    fig = plt.figure(figsize=(35, 10))  # Much larger figure
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1])
    
    for method_idx, (method, label) in enumerate(zip(methods, method_labels)):
        # Find a folder with resolution=100 and agents=30 for this method
        pattern = f"{method}_resolution=100_agents=30"
        matching_folders = [f for f in os.listdir(test_folder) 
                          if pattern in f and not f.endswith('.meta')]
        
        ax = plt.Subplot(fig, gs[method_idx])
        fig.add_subplot(ax)
        
        if matching_folders:
            folder = matching_folders[0]  # Take first matching folder
            screenshot_path = os.path.join(test_folder, folder, "step_1000.png")
            
            if os.path.exists(screenshot_path):
                img = plt.imread(screenshot_path)
                # Crop the image
                h, w = img.shape[:2]
                left = int(w * 0.25)
                right = int(w * 0.75)
                top = int(h * 0.10)
                bottom = int(h * 0.90)
                cropped_img = img[top:bottom, left:right]
                
                # Resize using PIL
                h_crop = bottom - top
                w_crop = right - left
                new_h = int(h_crop * 0.8)  # Made images a bit smaller
                new_w = int(w_crop * 0.8)
                
                cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
                resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                resized_img = np.array(resized_img) / 255.0
                
                ax.imshow(resized_img)
            else:
                ax.text(0.5, 0.5, "No screenshot found", ha='center', va='center', fontsize=24)  # Much bigger error text
        else:
            ax.text(0.5, 0.5, "No matching folder found", ha='center', va='center', fontsize=24)  # Much bigger error text
        
        ax.set_title(label, fontsize=28, pad=20, weight='bold')  # Much bigger titles with bold
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for bigger suptitle
    plt.savefig(os.path.join(results_folder, '4_screenshot_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test3_specific_screenshot_grid(data_folder, test_name):
    """Create screenshot grid for Test 3 with only force=400 sigma=12 combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Get all folders
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    print(f"Total folders found in Test 3: {len(all_folders)}")
    
    # Print a few example folder names for debugging
    print("First 5 folder names:")
    for i, folder in enumerate(all_folders[:5]):
        print(f"  {folder}")
    
    # Target parameters
    target_force = '400'  # Keep as string to match exactly
    target_sigma = '12'   # Keep as string to match exactly
    
    # Find folders with force=400 and sigma=12
    matching_folders = []
    for folder in all_folders:
        if f"force={target_force}" in folder and f"sigma={target_sigma}" in folder:
            matching_folders.append(folder)
    
    print(f"Found {len(matching_folders)} folders with force={target_force}, sigma={target_sigma}")
    
    if not matching_folders:
        # Let's see what force and sigma combinations actually exist
        print("Available force/sigma combinations:")
        combinations_found = set()
        for folder in all_folders:
            if "force=" in folder and "sigma=" in folder:
                try:
                    force_part = folder.split("force=")[1].split("_")[0]
                    sigma_part = folder.split("sigma=")[1].split("_")[0]
                    combinations_found.add(f"force={force_part}, sigma={sigma_part}")
                except:
                    pass
        
        for combo in sorted(combinations_found):
            print(f"  {combo}")
        
        # Create a placeholder image with bigger text
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # Bigger placeholder
        ax.text(0.5, 0.5, f"No data found for\nForce={target_force}, Sigma={target_sigma}", 
                ha='center', va='center', fontsize=24, weight='bold')  # Much bigger error text
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(results_folder, '3_vision_variations_grid.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Extract unique FOV and vision length combinations from matching folders
    combinations = []
    for folder in matching_folders:
        try:
            # Parse folder name to extract parameters
            # Example: "Vision_arcs=5_first=10_last=20_visionLength=30_fov=120_force=400_sigma=12_index=1"
            parts = folder.split('_')
            fov = None
            vision_length = None
            
            for part in parts:
                if part.startswith('fov='):
                    fov = part.split('=')[1]
                elif part.startswith('visionLength='):
                    vision_length = part.split('=')[1]
            
            if fov is not None and vision_length is not None:
                combo_key = f"fov={fov}_visionLength={vision_length}"
                # Check if we already have this combination
                if not any(c['key'] == combo_key for c in combinations):
                    combinations.append({
                        'fov': fov,
                        'vision_length': vision_length,
                        'folder': folder,
                        'key': combo_key,
                        'label': f'FOV={fov}°\nVision Length={vision_length}m'
                    })
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue
    
    if not combinations:
        print("No valid parameter combinations found")
        return
    
    # Sort combinations by FOV and vision length
    combinations.sort(key=lambda x: (int(x['fov']), int(x['vision_length'])))
    
    print(f"Found {len(combinations)} unique FOV/vision length combinations:")
    for combo in combinations:
        print(f"  FOV={combo['fov']}, Vision Length={combo['vision_length']}")
    
    # Create grid layout - much larger figure
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(28, 16))  # Much larger figure
    axes = axes.flatten()
    
    for idx, combo in enumerate(combinations):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        folder = combo['folder']
        screenshot_path = os.path.join(test_folder, folder, "step_1000.png")  # Using step_1000 since simulation steps = 1000
        
        if os.path.exists(screenshot_path):
            img = plt.imread(screenshot_path)
            # Crop the image
            h, w = img.shape[:2]
            left = int(w * 0.25)
            right = int(w * 0.75)
            top = int(h * 0.10)
            bottom = int(h * 0.90)
            cropped_img = img[top:bottom, left:right]
            ax.imshow(cropped_img)
        else:
            ax.text(0.5, 0.5, "No screenshot found", ha='center', va='center', fontsize=20)  # Bigger error text
        
        ax.set_title(combo['label'], fontsize=24, pad=15, weight='bold')  # Much bigger titles with bold
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(combinations), len(axes)):
        axes[idx].set_visible(False)

    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for bigger suptitle
    plt.savefig(os.path.join(results_folder, '3_vision_variations_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test3(data_folder):
    """Plot results for Vision Method Sampling Analysis"""
    test_name = "3_Sample_points_paired_with_forces_and_sigmas_small"
    
    print("Starting 3_vision_variations_grid...")
    create_test3_specific_screenshot_grid(data_folder, test_name)
    print("Completed 3_vision_variations_grid")

def create_performance_by_resolution(data_folder, test_name, methods, method_labels):
    """Create bar graph showing performance by resolution"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    
    fig, ax = plt.subplots(figsize=(18, 10))  # Larger figure
    
    bar_width = 0.15
    x = np.arange(len(resolutions))
    
    # Use distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        mins = []
        maxs = []
        for res in resolutions:
            folders = [f for f in os.listdir(test_folder) 
                      if method in f and f'resolution={res}' in f]
            run_means = []
            for folder in folders:
                data_path = os.path.join(test_folder, folder, 'data.csv')
                if os.path.exists(data_path):
                    df, _ = load_data(data_path)
                    if df is not None:
                        run_means.append(df['TimeForInterval'].mean() * 10)  # Multiply by 10
            
            if run_means:
                means.append(np.mean(run_means))
                mins.append(np.min(run_means))
                maxs.append(np.max(run_means))
            else:
                means.append(0)
                mins.append(0)
                maxs.append(0)
        
        yerr = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]
        ax.bar(x + i*bar_width, means, bar_width, label=label, color=color)
        ax.errorbar(x + i*bar_width, means, yerr=yerr, fmt='none', color='black', capsize=5)
    
    ax.set_xlabel('Resolution', fontsize=20)  # Bigger
    ax.set_ylabel('Average Time per Step (ms)', fontsize=20)  # Bigger
    ax.set_title('Performance by Resolution', fontsize=24)  # Bigger
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(resolutions, fontsize=18)  # Bigger
    ax.tick_params(axis='both', which='major', labelsize=18)  # Bigger
    ax.yaxis.set_major_locator(MultipleLocator(50))  # Reduce y-axis steps
    ax.legend(fontsize=16)  # Bigger
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '4_performance_by_resolution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test4_full_screenshot_grid(data_folder, test_name, methods, method_labels):
    """Create single grid of screenshots for all methods, resolutions and agent counts"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Get all resolutions and agent counts
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in all_folders if 'agents=' in f))
    
    # Create a grid with methods side by side for each resolution-agent combination
    fig = plt.figure(figsize=(30, 10))  # Larger figure
    outer_grid = gridspec.GridSpec(len(agent_counts), len(resolutions))
    
    for i, agents in enumerate(agent_counts):
        for j, res in enumerate(resolutions):
            # Create inner grid for methods
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, len(methods), 
                subplot_spec=outer_grid[i, j], wspace=0.1, hspace=0.1)
            
            for method_idx, (method, method_label) in enumerate(zip(methods, method_labels)):
                ax = plt.Subplot(fig, inner_grid[method_idx])
                fig.add_subplot(ax)
                
                pattern = f"{method}_resolution={res}_agents={agents}"
                matching_folders = [f for f in os.listdir(test_folder) 
                                  if pattern in f and not f.endswith('.meta')]
                
                if matching_folders:
                    folder = matching_folders[0]
                    screenshot_path = os.path.join(test_folder, folder, "step_1000.png")
                    
                    if os.path.exists(screenshot_path):
                        img = plt.imread(screenshot_path)
                        # Crop the image
                        h, w = img.shape[:2]
                        left = int(w * 0.25)
                        right = int(w * 0.75)
                        top = int(h * 0.10)
                        bottom = int(h * 0.90)
                        cropped_img = img[top:bottom, left:right]
                        
                        # Resize using PIL
                        h_crop = bottom - top
                        w_crop = right - left
                        new_h = int(h_crop * 0.5)  # Made images smaller
                        new_w = int(w_crop * 0.5)
                        
                        cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
                        resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        resized_img = np.array(resized_img) / 255.0
                        
                        ax.imshow(resized_img)
                    else:
                        ax.text(0.5, 0.5, "No screenshot", ha='center', va='center', fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=8)
                
                # Add labels with bigger text
                if i == 0 and method_idx == len(methods)//2:  # Top row, middle method
                    ax.set_title(f'Resolution {res}', fontsize=14)  # Bigger
                if j == 0 and method_idx == 0:  # Left column, first method
                    ax.set_ylabel(f'{agents} Agents', fontsize=14)  # Bigger
                
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.suptitle('Trail Formation Grid\nDisplay order per group: Helbing, Vision (30 pts), Vision (75 pts), Vision (200 pts), Vision (1000 pts)', y=0.98, fontsize=16)  # Bigger
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '4_screenshot_grid_full.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_folder = "../ExperimentData"
    if not os.path.exists(data_folder):
        print(f"Creating data folder: {os.path.abspath(data_folder)}")
        os.makedirs(data_folder)
    
    plot_test1(data_folder)
    plot_test2(data_folder)
    plot_test3(data_folder)
    plot_test4(data_folder)