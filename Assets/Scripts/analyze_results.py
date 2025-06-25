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
    
    # Create distinct color scheme following Simulator color progression: Red → Yellow → Green → Cyan → Blue → Magenta → Pink
    # Use 7 distinct color families for better differentiation, avoiding orange
    import matplotlib.colors as mcolors
    
    # Define custom color families based on simulator colors - using explicit color values
    def create_single_color_cmap(color):
        """Create a colormap from light to dark version of the color"""
        return mcolors.LinearSegmentedColormap.from_list("", ["#F0F0F0", color])
    
    T_color_families = [
        create_single_color_cmap('#FF0000'),    # T 1-3: Red
        create_single_color_cmap('#FFFF00'),    # T 4-6: Yellow  
        create_single_color_cmap('#00FF00'),    # T 7-9: Green
        create_single_color_cmap('#00FFFF'),    # T 10-12: Cyan
        create_single_color_cmap('#0000FF'),    # T 13-15: Blue
        create_single_color_cmap('#FF00FF'),    # T 16-18: Magenta
        create_single_color_cmap('#FF0080')     # T 19-20: Pink
    ]
    
    # Define line styles for different I ranges
    I_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, densely dashdotdotted
    
    # Store line styles
    line_styles = {}
    
    # Pre-calculate all styles 
    for T in range(1, 21):
        for I in range(1, 21):
            # Determine color family based on T value (0-6 index for 7 families)
            T_family_idx = min(6, (T-1) // 3)  # Groups of 3: 1-3, 4-6, 7-9, 10-12, 13-15, 16-18, 19-20
            color_family = T_color_families[T_family_idx]
            
            # Get color intensity within family based on T position within group
            T_within_group = ((T-1) % 3) / 3.0  # 0.0 to 0.67 for groups of 3
            color_intensity = 0.4 + T_within_group * 0.6  # 0.4 to 1.0
            T_color = color_family(color_intensity)
            
            # Determine linestyle based on I value (0-4 index) 
            I_style_idx = min(4, (I-1) // 4)
            linestyle = I_linestyles[I_style_idx]
            
            # Alpha based on I value - higher I = more opaque
            alpha = 0.3 + (I-1) * 0.7 / 19  # 0.3 to 1.0
            
            line_styles[(T,I)] = {
                'color': T_color, 
                'alpha': alpha,
                'linestyle': linestyle
            }
    
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
                                   **line_styles[(T,I)], linewidth=2)  # Thicker lines
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

    # Color guide removed - explanation will be in thesis text
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
    
    # Create distinct color scheme following Simulator color progression: Red → Yellow → Green → Cyan → Blue → Magenta → Pink
    # Use 7 distinct color families for better differentiation, avoiding orange
    import matplotlib.colors as mcolors
    
    # Define custom color families based on simulator colors - using explicit color values
    def create_single_color_cmap(color):
        """Create a colormap from light to dark version of the color"""
        return mcolors.LinearSegmentedColormap.from_list("", ["#F0F0F0", color])
    
    T_color_families = [
        create_single_color_cmap('#FF0000'),    # T 1-3: Red
        create_single_color_cmap('#FFFF00'),    # T 4-6: Yellow  
        create_single_color_cmap('#00FF00'),    # T 7-9: Green
        create_single_color_cmap('#00FFFF'),    # T 10-12: Cyan
        create_single_color_cmap('#0000FF'),    # T 13-15: Blue
        create_single_color_cmap('#FF00FF'),    # T 16-18: Magenta
        create_single_color_cmap('#FF0080')     # T 19-20: Pink
    ]
    
    # Define line styles for different I ranges
    I_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, densely dashdotdotted
    
    line_styles = {}
    
    for T in range(1, 21):
        for I in range(1, 21):
            # Determine color family based on T value (0-6 index for 7 families)
            T_family_idx = min(6, (T-1) // 3)  # Groups of 3: 1-3, 4-6, 7-9, 10-12, 13-15, 16-18, 19-20
            color_family = T_color_families[T_family_idx]
            
            # Get color intensity within family based on T position within group
            T_within_group = ((T-1) % 3) / 3.0  # 0.0 to 0.67 for groups of 3
            color_intensity = 0.4 + T_within_group * 0.6  # 0.4 to 1.0
            T_color = color_family(color_intensity)
            
            # Determine linestyle based on I value (0-4 index) 
            I_style_idx = min(4, (I-1) // 4)
            linestyle = I_linestyles[I_style_idx]
            
            # Alpha based on I value - higher I = more opaque
            alpha = 0.3 + (I-1) * 0.7 / 19  # 0.3 to 1.0
            
            line_styles[(T,I)] = {
                'color': T_color, 
                'alpha': alpha,
                'linestyle': linestyle
            }

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_count = 0
        
        for (T, I) in data_paths:
            result = load_data(data_paths[(T, I)])
            if result:
                data_df, _ = result
                if len(data_df) >= 20:
                    steps = range(100, 2100, 100)
                    ax.plot(steps, data_df[metric], **line_styles[(T,I)], linewidth=2)  # Thicker lines
                    plotted_count += 1
        
        print(f"\nPlotted {plotted_count} lines for {metric}")
        
        ax.set_title(f'{metric} over Time (Efficiency > 60)', fontsize=18)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, 2000)

    # Color guide removed - explanation will be in thesis text
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '1_metrics_time_series_with_Efficiency_over_60.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_specific_time_series(data_folder, test_name, metrics):
    """Create time series plots for specific T,I combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Specific combinations to show
    combinations = [
        {'T': 9, 'I': 6, 'label': 'Well spread out agents (T=9, I=6)'},
        {'T': 9, 'I': 7, 'label': 'Small change, big difference (T=9, I=7)'},
        {'T': 11, 'I': 16, 'label': 'Extreme case (T=11, I=16)'},
        {'T': 18, 'I': 7, 'label': 'Another extreme case (T=18, I=7)'}
    ]
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 8*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    # Create distinct colors for each combination
    colors = ['#2E8B57', '#FF6347', '#4169E1', '#FF8C00']  # Sea Green, Tomato, Royal Blue, Dark Orange
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for combo_idx, combo in enumerate(combinations):
            T, I = combo['T'], combo['I']
            expected_index = ((T-1) * 20) + I
            expected_folder = f"Helbing_T={T}_I={I}_repetition=1_index={expected_index}"
            
            # Find matching folder
            matching_folders = [f for f in os.listdir(test_folder) 
                              if expected_folder in f and not f.endswith('.meta')]
            
            if matching_folders:
                folder = matching_folders[0]
                data_path = os.path.join(test_folder, folder, "data.csv")
                
                if os.path.exists(data_path):
                    result = load_data(data_path)
                    if result:
                        data_df, _ = result
                        if len(data_df) >= 20:
                            steps = range(100, 2100, 100)
                            ax.plot(steps, data_df[metric], 
                                   color=colors[combo_idx], 
                                   linewidth=3, 
                                   label=combo['label'],
                                   marker='o', 
                                   markersize=4)
                        else:
                            print(f"✗ Not enough data points for: {combo['label']}")
                else:
                    print(f"✗ Missing data file for: {combo['label']}")
            else:
                print(f"✗ Missing folder for: {combo['label']}")
        
        ax.set_title(f'{metric} over Time - Specific Cases', fontsize=18)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=12, loc='best')
        ax.set_xlim(0, 2000)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '1_specific_cases_time_series.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_specific_screenshot_grid(data_folder, test_name):
    """Create a focused screenshot grid with only specific T,I combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Specific combinations to show
    combinations = [
        {'T': 9, 'I': 6, 'label': 'Well spread out agents\n(T=9, I=6)'},
        {'T': 9, 'I': 7, 'label': 'Small change, big difference\n(T=9, I=7)'},
        {'T': 11, 'I': 16, 'label': 'Extreme case\n(T=11, I=16)'},
        {'T': 18, 'I': 7, 'label': 'Another extreme case\n(T=18, I=7)'}
    ]
    
    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(18, 12))
    
    # Create 2x2 grid for images with some space for colorbar
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.15], wspace=0.3)
    
    # Create subplot for each combination
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            combo = combinations[idx]
            ax = plt.Subplot(fig, gs[i, j])
            fig.add_subplot(ax)
            
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
    
    # Create colorbar on the right side spanning both rows
    cbar_ax = plt.Subplot(fig, gs[:, 2])
    fig.add_subplot(cbar_ax)
    
    # Create comfort map colorbar based on Unity gradient
    comfort_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', '#ff0080']  # pink
    comfort_levels = [0, 4, 8, 12, 16, 18, 20]
    
    # Create a custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('comfort', comfort_colors, N=256)
    
    # Create gradient data for visualization
    gradient = np.linspace(0, 20, 256).reshape(256, 1)
    
    # Display the gradient
    cbar_ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 20])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 20)
    
    # Set ticks and labels
    cbar_ax.set_yticks(comfort_levels)
    cbar_ax.set_yticklabels(comfort_levels, fontsize=12)
    cbar_ax.set_xticks([])
    cbar_ax.set_ylabel('Comfort Level', fontsize=14, rotation=270, labelpad=20)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.yaxis.tick_right()
    
    plt.savefig(os.path.join(results_folder, '1_screenshot_grid_mini.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test1_comprehensive_screenshot_grid(data_folder, test_name):
    """Create comprehensive screenshot grid for Test 1 with all T,I combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
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
                   cmap='Blues' if metric == 'Civility' else 'Reds',
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
                    
                    # Find matching folders with this force and sigma combination
                    pattern = f"{method}_force={force_str}_sigma={sigma}_index="
                    values = []
                    
                    # Look for all folders matching this pattern
                    for folder in os.listdir(test_folder):
                        if pattern in folder and not folder.endswith('.meta'):
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
                       cmap='Blues' if metric == 'Civility' else 'Reds',
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
    
    # Specific Helbing combinations to show
    combinations = [
        {'force': 2, 'sigma': 7, 'label': 'Good trails\n(F=2, σ=7)'},
        {'force': 0.5, 'sigma': 7, 'label': 'Too little force\n(F=0.5, σ=7)'},
        {'force': 2, 'sigma': 4, 'label': 'Too little sigma\n(F=2, σ=4)'},
        {'force': 2.5, 'sigma': 4, 'label': 'Pooling behavior\n(F=2.5, σ=4)'},
        {'force': 3, 'sigma': 10, 'label': 'Pooling behavior\n(F=3, σ=10)'}
    ]
    
    # Create figure with space for colorbar
    fig = plt.figure(figsize=(38, 8))
    
    # Create layout with screenshots and colorbar
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.15], wspace=0.4)
    
    for idx, combo in enumerate(combinations):
        ax = plt.Subplot(fig, gs[0, idx])
        fig.add_subplot(ax)
        
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
    
    # Create colorbar
    cbar_ax = plt.Subplot(fig, gs[0, 5])
    fig.add_subplot(cbar_ax)
    
    # Create comfort map colorbar based on Unity gradient
    comfort_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', '#ff0080']  # pink
    comfort_levels = [0, 4, 8, 12, 16, 18, 20]
    
    # Create a custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('comfort', comfort_colors, N=256)
    
    # Create gradient data for visualization
    gradient = np.linspace(0, 20, 256).reshape(256, 1)
    
    # Display the gradient
    cbar_ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 20])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 20)
    
    # Set ticks and labels
    cbar_ax.set_yticks(comfort_levels)
    cbar_ax.set_yticklabels(comfort_levels, fontsize=12)
    cbar_ax.set_xticks([])
    cbar_ax.set_ylabel('Comfort Level', fontsize=14, rotation=270, labelpad=20)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.yaxis.tick_right()
    
    plt.savefig(os.path.join(results_folder, '2_parameter_grid_small.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_comprehensive_screenshot_grid(data_folder, test_name):
    """Create comprehensive screenshot grid for Test 2 with all parameter combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
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
    """Create time series plots comparing both methods with improved color scheme"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9.6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    # Define parameter ranges for consistent styling
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
    
    # Create color families for different methods using simulator color progression
    # Helbing: Red → Yellow → Green, Vision: Cyan → Blue → Magenta → Pink
    import matplotlib.colors as mcolors
    
    def create_single_color_cmap(color):
        """Create a colormap from light to dark version of the color"""
        return mcolors.LinearSegmentedColormap.from_list("", ["#F0F0F0", color])
    
    method_color_families = {
        'Helbing': [
            create_single_color_cmap('#FF0000'),    # Red for low forces
            create_single_color_cmap('#FFFF00'),    # Yellow for mid forces  
            create_single_color_cmap('#00FF00')     # Green for high forces
        ],
        'Vision': [
            create_single_color_cmap('#00FFFF'),    # Cyan for low forces
            create_single_color_cmap('#0000FF'),    # Blue for mid forces
            create_single_color_cmap('#FF00FF'),    # Magenta for high forces
            create_single_color_cmap('#FF0080')     # Pink for highest forces
        ]
    }
    
    # Define line styles for sigma ranges (sigma determines line style)
    sigma_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, densely dashdotdotted
    
    # Pre-calculate line styles for each method
    line_styles = {}
    
    for method in methods:
        force_range = param_ranges[method]['forces']
        sigma_range = param_ranges[method]['sigmas']
        
        # Group forces based on available color families (3 for Helbing, 4 for Vision)
        force_groups = len(method_color_families[method])
        forces_per_group = len(force_range) // force_groups
        
        # Group sigmas into 5 ranges for line styles  
        sigma_groups = min(5, len(sigma_range))
        sigmas_per_group = max(1, len(sigma_range) // sigma_groups)
        
        for i, force in enumerate(force_range):
            for j, sigma in enumerate(sigma_range):
                # Determine color family based on force position
                force_group_idx = min(force_groups - 1, i // max(1, forces_per_group))
                color_family = method_color_families[method][force_group_idx]
                
                # Get color intensity within family based on force position within group
                force_within_group = (i % max(1, forces_per_group)) / max(1, forces_per_group - 1) if forces_per_group > 1 else 0
                color_intensity = 0.4 + force_within_group * 0.6  # 0.4 to 1.0
                force_color = color_family(color_intensity)
                
                # Determine line style based on sigma position
                sigma_group_idx = min(sigma_groups - 1, j // max(1, sigmas_per_group))
                linestyle = sigma_linestyles[sigma_group_idx]
                
                # Alpha based on sigma value - higher sigma = more opaque
                sigma_min, sigma_max = min(sigma_range), max(sigma_range)
                if sigma_max > sigma_min:
                    alpha = 0.3 + (sigma - sigma_min) * 0.7 / (sigma_max - sigma_min)
                else:
                    alpha = 0.6
                
                line_styles[(method, force, sigma)] = {
                    'color': force_color,
                    'alpha': alpha,
                    'linestyle': linestyle
                }
    
    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_count = 0
        
        # Plot data for each method
        for method in methods:
            # Get all matching folders for this method
            folders = [f for f in os.listdir(test_folder) 
                      if f.startswith(method) and not f.endswith('.meta')]
            
            # Plot each configuration
            for folder in folders:
                data_path = os.path.join(test_folder, folder, "data.csv")
                if os.path.exists(data_path):
                    result = load_data(data_path)
                    if result:
                        data_df, _ = result
                        if len(data_df) >= 20:
                            # Extract force and sigma from folder name
                            parts = folder.split('_')
                            try:
                                force = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('force='))
                                sigma = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('sigma='))
                                
                                # Use pre-calculated line style if available
                                style_key = (method, force, sigma)
                                if style_key in line_styles:
                                    ax.plot(data_df['Step'], data_df[metric],
                                           **line_styles[style_key], linewidth=2)
                                    plotted_count += 1
                                         
                            except Exception as e:
                                print(f"Error processing folder {folder}: {e}")
                                continue
        
        print(f"\nPlotted {plotted_count} lines for {metric}")
        
        ax.set_title(f'{metric} over Time', fontsize=18)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
    
    # Add a color guide as text in the corner of the first subplot
    # Color guide removed - explanation will be in thesis text
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '2_combined_time_series.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_similar_combinations_time_series(data_folder, test_name):
    """Create time series plots for the same combinations as the screenshots"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
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
                    result = load_data(data_path)
                    if result:
                        data_df, _ = result
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
    
    print("Starting 1_specific_cases_time_series...")
    create_test1_specific_time_series(data_folder, test_name, metrics)
    print("Completed 1_specific_cases_time_series")

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
    
    #print("Starting 4_screenshot_grid_full...")
    #create_test4_full_screenshot_grid(data_folder, test_name,
    #                                methods=methods,
    #                                method_labels=method_labels)
    #print("Completed 4_screenshot_grid_full")

def create_resolution_agent_heatmap(data_folder, test_name, metrics, method, filename):
    """Create heatmap of metrics vs resolution and agent count"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
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
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    metrics = ['Efficiency', 'Civility', 'TimeForInterval']
    metric_labels = ['Efficiency', 'Civility', 'Time per Step (ms)']
    
    # Improved color scheme for Test 4
    method_colors = [
        '#e31a1c',  # Red for Helbing
        '#1f78b4',  # Blue for Vision (30 pts)
        '#33a02c',  # Green for Vision (75 pts)  
        '#ff7f00',  # Orange for Vision (200 pts)
        '#6a3d9a'   # Purple for Vision (1000 pts)
    ]
    
    # Distinct line styles for better readability
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx]
        
        for method_idx, (method, label) in enumerate(zip(methods, method_labels)):
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
                        # For TimeForInterval, multiply by 10 to get ms
                        if metric == 'TimeForInterval':
                            df_copy = df.copy()
                            df_copy[metric] = df_copy[metric] * 10
                            dfs.append(df_copy[['Step', metric]])
                        else:
                            dfs.append(df[['Step', metric]])
            
            if dfs:
                # Calculate mean and std
                mean_df = pd.concat(dfs).groupby('Step').mean()
                std_df = pd.concat(dfs).groupby('Step').std()
                
                # Plot mean line with confidence interval
                ax.plot(mean_df.index, mean_df[metric], 
                       label=label, 
                       color=method_colors[method_idx],
                       linestyle=linestyles[method_idx],
                       linewidth=2.5,
                       alpha=0.8)
                ax.fill_between(mean_df.index,
                              mean_df[metric] - std_df[metric],
                              mean_df[metric] + std_df[metric],
                              color=method_colors[method_idx],
                              alpha=0.15)
        
        ax.set_xlabel('Step', fontsize=14, weight='bold')
        ax.set_ylabel(metric_label, fontsize=14, weight='bold')
        ax.set_title(f'{metric_label} over Time', fontsize=16, weight='bold', pad=15)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_bar_graph(data_folder, test_name, methods, method_labels, filename):
    """Create single bar graph showing all performance data with error bars"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in all_folders if 'agents=' in f))
    
    combinations = [(res, agents) for res in resolutions for agents in agent_counts]
    
    fig, ax = plt.subplots(figsize=(24, 10))  # Larger figure
    
    bar_width = 0.15
    x = np.arange(len(combinations))
    
    # Improved color scheme for Test 4 - distinct and professional
    method_colors = [
        '#e31a1c',  # Red for Helbing
        '#1f78b4',  # Blue for Vision (30 pts)
        '#33a02c',  # Green for Vision (75 pts)  
        '#ff7f00',  # Orange for Vision (200 pts)
        '#6a3d9a'   # Purple for Vision (1000 pts)
    ]
    
    for i, (method, label) in enumerate(zip(methods, method_labels)):
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
        bars = ax.bar(x + i*bar_width, means, bar_width, 
                     label=label, color=method_colors[i], 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + i*bar_width, means, yerr=yerr, 
                   fmt='none', color='black', capsize=3, capthick=1.5, alpha=0.7)
    
    ax.set_xlabel('Resolution-Agent Count Combinations', fontsize=20, weight='bold')
    ax.set_ylabel('Average Time per Step (ms)', fontsize=20, weight='bold')
    ax.set_title('Performance by Resolution and Agent Count', fontsize=24, weight='bold', pad=20)
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels([f'R{res}-A{agents}' for res, agents in combinations], 
                      rotation=45, fontsize=14, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.legend(fontsize=16, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
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
    
    # Improved color scheme for Test 4 - distinct and professional
    method_colors = [
        '#e31a1c',  # Red for Helbing
        '#1f78b4',  # Blue for Vision (30 pts)
        '#33a02c',  # Green for Vision (75 pts)  
        '#ff7f00',  # Orange for Vision (200 pts)
        '#6a3d9a'   # Purple for Vision (1000 pts)
    ]
    
    for i, (method, label) in enumerate(zip(methods, method_labels)):
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
        bars = ax.bar(x + i*bar_width, means, bar_width, 
                     label=label, color=method_colors[i], 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + i*bar_width, means, yerr=yerr, 
                   fmt='none', color='black', capsize=3, capthick=1.5, alpha=0.7)
    
    ax.set_xlabel('Agent Count', fontsize=20, weight='bold')
    ax.set_ylabel('Average Time per Step (ms)', fontsize=20, weight='bold')
    ax.set_title('Performance at Resolution 700', fontsize=24, weight='bold', pad=20)
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(agent_counts, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.legend(fontsize=16, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
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
    
    # Improved color scheme and line styles for Test 4 time series
    method_colors = [
        '#e31a1c',  # Red for Helbing
        '#1f78b4',  # Blue for Vision (30 pts)
        '#33a02c',  # Green for Vision (75 pts)  
        '#ff7f00',  # Orange for Vision (200 pts)
        '#6a3d9a'   # Purple for Vision (1000 pts)
    ]
    
    # Distinct line styles for better readability
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, densely dashdotted
    
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
                           color=method_colors[method_idx],
                           alpha=0.8,
                           linewidth=2.5,  # Thicker lines
                           marker='o' if method_idx == 0 else None,  # Add markers only for Helbing
                           markersize=4,
                           markevery=10)  # Show every 10th marker
        
        ax.set_xlabel('Step', fontsize=20, weight='bold')
        ax.set_ylabel('Time per Step (ms)', fontsize=20, weight='bold')
        ax.set_title(f'Computation Time per Step (Resolution {resolution})', fontsize=24, weight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=16, frameon=True, fancybox=True, shadow=True)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MultipleLocator(100))  # Set y-axis ticks every 100ms
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'4_time_series_res_{resolution}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_test4_screenshot_grid(data_folder, test_name, methods, method_labels):
    """Create grid of final screenshots for each configuration"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Create a 1x6 grid (5 methods + 1 colorbar) - much larger figure
    fig = plt.figure(figsize=(40, 10))  # Increased width for colorbar
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.15], wspace=0.3)
    
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

    # Add comfort map colorbar on the right side
    cbar_ax = plt.Subplot(fig, gs[5])
    fig.add_subplot(cbar_ax)
    
    # Create comfort colormap (0 = red, 20 = pink)
    comfort_colormap = create_comfort_colormap()
    
    # Create a gradient for the colorbar
    gradient = np.linspace(0, 20, 256).reshape(256, 1)
    cbar_ax.imshow(gradient, aspect='auto', cmap=comfort_colormap, extent=[0, 1, 0, 20])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 20)
    cbar_ax.set_yticks([0, 4, 8, 12, 16, 20])
    cbar_ax.set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=20)
    cbar_ax.set_xticks([])
    cbar_ax.set_ylabel('Comfort Level', fontsize=24, weight='bold')
    cbar_ax.set_title('Comfort\nMap', fontsize=22, weight='bold', pad=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for bigger suptitle
    plt.savefig(os.path.join(results_folder, '4_screenshot_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test3_specific_screenshot_grid(data_folder, test_name):
    """Create screenshot grid for Test 3 with only force=400 sigma=12 combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
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
    
    # Create grid layout with space for colorbar
    fig = plt.figure(figsize=(32, 16))
    
    # Create grid with space for colorbar - 2 rows, 4 columns (3 for images, 1 for colorbar)
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.15], wspace=0.3, hspace=0.3)
    
    for idx, combo in enumerate(combinations):
        if idx >= 6:  # Limit to 6 combinations (2 rows x 3 cols)
            break
            
        row = idx // 3
        col = idx % 3
        ax = plt.Subplot(fig, gs[row, col])
        fig.add_subplot(ax)
        
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
    
    # Create colorbar on the right side spanning both rows
    cbar_ax = plt.Subplot(fig, gs[:, 3])
    fig.add_subplot(cbar_ax)
    
    # Create comfort map colorbar based on Unity gradient
    comfort_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', '#ff0080']  # pink
    comfort_levels = [0, 4, 8, 12, 16, 18, 20]
    
    # Create a custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('comfort', comfort_colors, N=256)
    
    # Create gradient data for visualization
    gradient = np.linspace(0, 20, 256).reshape(256, 1)
    
    # Display the gradient
    cbar_ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 20])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 20)
    
    # Set ticks and labels
    cbar_ax.set_yticks(comfort_levels)
    cbar_ax.set_yticklabels(comfort_levels, fontsize=12)
    cbar_ax.set_xticks([])
    cbar_ax.set_ylabel('Comfort Level', fontsize=14, rotation=270, labelpad=20)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.yaxis.tick_right()
    
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
    
    # Improved color scheme for Test 4 - distinct and professional
    method_colors = [
        '#e31a1c',  # Red for Helbing
        '#1f78b4',  # Blue for Vision (30 pts)
        '#33a02c',  # Green for Vision (75 pts)  
        '#ff7f00',  # Orange for Vision (200 pts)
        '#6a3d9a'   # Purple for Vision (1000 pts)
    ]
    
    for i, (method, label) in enumerate(zip(methods, method_labels)):
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
        bars = ax.bar(x + i*bar_width, means, bar_width, 
                     label=label, color=method_colors[i], 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + i*bar_width, means, yerr=yerr, 
                   fmt='none', color='black', capsize=3, capthick=1.5, alpha=0.7)
    
    ax.set_xlabel('Resolution', fontsize=20, weight='bold')
    ax.set_ylabel('Average Time per Step (ms)', fontsize=20, weight='bold')
    ax.set_title('Performance by Resolution', fontsize=24, weight='bold', pad=20)
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(resolutions, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.legend(fontsize=16, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '4_performance_by_resolution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test4_full_screenshot_grid(data_folder, test_name, methods, method_labels):
    """Create single grid of screenshots for all methods, resolutions and agent counts"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
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
    
    #plot_test1(data_folder)
    #plot_test2(data_folder)
    #plot_test3(data_folder)
    plot_test4(data_folder)