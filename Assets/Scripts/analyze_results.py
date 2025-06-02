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
    
    # Create figure with taller plots (20% taller)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(30, 9.6*len(metrics)))  # Changed from 8 to 9.6
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
    
    # Store line styles and handles for legend
    line_styles = {}
    legend_handles = []
    legend_labels = []
    
    # Pre-calculate all colors and sort by T value
    for T in range(1, 21):
        for I in range(1, 21):
            T_color = np.array(matplotlib.colors.to_rgba(T_colors[T-1]))
            I_color = I_colors[I-1]
            blended_color = (T_color + I_color) / 2
            line_styles[(T,I)] = {'color': blended_color, 'alpha': 0.7}  # Increased alpha for better visibility
    
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
                            line = ax.plot(steps, data_df[metric], 
                                         **line_styles[(T,I)])
                            
                            # Only collect legend handles from first plot
                            if metric_idx == 0:
                                legend_handles.append(line[0])
                                legend_labels.append(f'T={T},I={I}')
                            
                            plotted_count += 1
                        else:
                            print(f"✗ Not enough data points in: {expected_folder}")
                else:
                    print(f"✗ Missing folder: {expected_folder}")
        
        print(f"\nPlotted {plotted_count} lines for {metric}")
        
        ax.set_title(f'{metric} over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.grid(True)
        
        # Set the expanded y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # Set x-axis to full range (0 to 2000)
        ax.set_xlim(0, 2000)
    
    # Add single legend between the plots on the right side
    fig.legend(legend_handles, legend_labels,
              bbox_to_anchor=(0.8, 0.5),
              loc='center left',
              ncol=4,
              borderaxespad=0)
    
    # Adjust subplot parameters
    plt.subplots_adjust(
        right=0.75,
        hspace=0.1,
        top=1.02,
        bottom=-0.02
    )
    
    # Adjust height based on legend size
    adjust_plot_height(axes[0], fig, 2)
    
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

    # Create figure and axes
    fig, axes = plt.subplots(len(metrics), 1, figsize=(30, 9.6*len(metrics)))  # Changed from 8 to 9.6
    if len(metrics) == 1:
        axes = [axes]
    
    # Create distinct colors for T values
    T_colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    I_colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    line_styles = {}
    legend_handles = []
    legend_labels = []
    
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
                    line = ax.plot(steps, data_df[metric], **line_styles[(T,I)])
                    if metric_idx == 0:
                        legend_handles.append(line[0])
                        legend_labels.append(f'T={T},I={I}')
                    plotted_count += 1
        
        print(f"\nPlotted {plotted_count} lines for {metric}")
        
        ax.set_title(f'{metric} over Time (Efficiency > 60)')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.set_xlim(0, 2000)

    fig.legend(legend_handles, legend_labels,
              bbox_to_anchor=(0.8, 0.5),
              loc='center left',
              ncol=1,
              borderaxespad=0)
    
    plt.subplots_adjust(right=0.75, hspace=0.1, top=0.80, bottom=0.2)
    plt.savefig(os.path.join(results_folder, '1_metrics_time_series_with_Efficiency_over_60.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_screenshot_grid(data_folder, test_name, pattern_groups, title, filename):
    """Create a grid of screenshots from simulation runs"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # For Test 2, create a grid based on force and sigma values for each method
    if "2_Force_and_Sigma" in test_name:
        # Store folder paths for each force/sigma combination
        method_data = {}
        for method in ['Helbing', 'Vision']:
            forces = set()
            sigmas = set()
            screenshot_paths = {}  # Store the actual paths
            
            folders = [f for f in os.listdir(test_folder) 
                      if method in f and 'repetition=1' in f and not f.endswith('.meta')]
            
            print(f"\nFound {len(folders)} folders for {method}")
            
            for folder in folders:
                parts = folder.split('_')
                try:
                    # Handle comma in force value
                    force_part = next(part for part in parts if part.startswith('force='))
                    force = float(force_part.split('=')[1].replace(',', '.'))
                    
                    sigma_part = next(part for part in parts if part.startswith('sigma='))
                    sigma = float(sigma_part.split('=')[1].replace(',', '.'))
                    
                    # Verify screenshot exists before adding parameters
                    screenshot_path = os.path.join(test_folder, folder, "step_2000.png")
                    if os.path.exists(screenshot_path):
                        forces.add(force)
                        sigmas.add(sigma)
                        # Store the path for this force/sigma combination
                        screenshot_paths[(force, sigma)] = screenshot_path
                        print(f"Found {method} screenshot for force={force} sigma={sigma}")
                except Exception as e:
                    print(f"Error processing folder {folder}: {e}")
                    continue
            
            method_data[method] = {
                'forces': sorted(forces),
                'sigmas': sorted(sigmas),
                'screenshots': screenshot_paths
            }
            print(f"Found {len(forces)} forces and {len(sigmas)} sigmas for {method} with valid screenshots")
        
        # Create subplot for each method
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        
        for method_idx, method in enumerate(['Helbing', 'Vision']):
            data = method_data[method]
            forces = data['forces']
            sigmas = data['sigmas']
            screenshots = data['screenshots']
            
            if not forces or not sigmas:
                print(f"No data found for {method}")
                continue
            
            print(f"Creating grid for {method} with {len(forces)} forces and {len(sigmas)} sigmas")
            
            # Create subplot grid for this method
            inner_gs = gridspec.GridSpecFromSubplotSpec(len(forces), len(sigmas), 
                                                      subplot_spec=gs[method_idx],
                                                      wspace=0.1, hspace=0.1)
            
            # Add method label
            plt.figtext(0.02, 0.75 - method_idx * 0.5, method, 
                       rotation=90, va='center', fontsize=12)
            
            # Create grid of screenshots
            for i, force in enumerate(forces):
                for j, sigma in enumerate(sigmas):
                    ax = plt.Subplot(fig, inner_gs[i, j])
                    fig.add_subplot(ax)
                    
                    screenshot_path = screenshots.get((force, sigma))
                    if screenshot_path and os.path.exists(screenshot_path):
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
                    print(f"No screenshot found for {method} force={force} sigma={sigma}")
                    ax.text(0.5, 0.5, "No image", ha='center', va='center')
                
                    # Add labels
                    if i == 0:  # Top row
                        ax.set_title(f'σ={sigma:.1f}')
                    if j == 0:  # Left column
                        ax.set_ylabel(f'f={force:.1f}')
                
                ax.set_xticks([])
                ax.set_yticks([])
    
        plt.suptitle(title, y=0.95)
        # Adjust spacing
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
        plt.savefig(os.path.join(results_folder, filename), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_heatmap(data_folder, test_name, metrics, methods, title="Combined Metrics Comparison"):
    """Create a 2x3 heatmap comparing methods"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    comfort_cmap = create_comfort_colormap()
    
    for metric_idx, metric in enumerate(metrics):
        for method_idx, method in enumerate(methods):
            ax = fig.add_subplot(gs[metric_idx, method_idx])
            
            # Collect data
            data = []
            for folder in os.listdir(test_folder):
                if method in folder:
                    result = load_data(os.path.join(test_folder, folder, "data.csv"))
                    if result:
                        data_df, _ = result
                        final_value = data_df[metric].iloc[-1]
                        data.append((folder, final_value))
            
            if data:
                # Create heatmap
                df = pd.DataFrame(data, columns=['folder', 'value'])
                pivot_table = create_pivot_table(df, method)
                
                if pivot_table is not None:
                    # Plot heatmap
                    sns.heatmap(pivot_table, ax=ax, 
                              cmap=comfort_cmap if metric == 'Civility' else 'viridis')
                    ax.set_title(f"{method} {metric}")
                else:
                    ax.text(0.5, 0.5, "No data available", 
                           ha='center', va='center')
                    ax.set_title(f"{method} {metric} (Error)")
        
        # Add difference plot if comparing methods
        if len(methods) > 1:
            ax = fig.add_subplot(gs[metric_idx, 2])
            diff_data = calculate_method_difference(test_folder, metric, methods)
            if diff_data is not None:
                sns.heatmap(diff_data, ax=ax, 
                           cmap='RdBu_r',
                           center=0)
                ax.set_title(f"Difference ({methods[0]} - {methods[1]})")
            else:
                ax.text(0.5, 0.5, "Could not calculate difference", 
                       ha='center', va='center')
                ax.set_title("Difference (Error)")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'combined_metrics_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison(data_folder, test_name, methods):
    """Create performance comparison plots"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Collect performance data
    performance_data = {}
    for method in methods:
        performance_data[method] = []
        for folder in os.listdir(test_folder):
            if method in folder:
                result = load_data(os.path.join(test_folder, folder, "data.csv"))
                if result:
                    data_df, _ = result
                    avg_time = data_df['TimeForInterval'].mean()
                    performance_data[method].append((folder, avg_time))
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, method in enumerate(methods):
        times = [t for _, t in performance_data[method]]
        ax.bar(x[i], np.mean(times), width, label=method)
        ax.errorbar(x[i], np.mean(times), yerr=np.std(times), color='black')
    
    ax.set_ylabel('Average Time per Step (ms)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_time_series(data_folder, test_name, metrics, methods, filename='2_combined_time_series.png'):
    """Create time series plots comparing both methods"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 12))
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
    
    # Store handles and labels for each method separately
    helbing_handles = []
    helbing_labels = []
    vision_handles = []
    vision_labels = []
    
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
                            
                            line = ax.plot(data_df['Step'], data_df[metric],
                                         color=combined_color, alpha=0.3,
                                         label=f'{method} (f={force}, σ={sigma})')
                            
                            # Store handles and labels for the legend
                            if metric_idx == 0:  # Only store once
                                if method == 'Helbing':
                                    helbing_handles.append(line[0])
                                    helbing_labels.append(f'H (f={force}, σ={sigma})')
                                else:
                                    vision_handles.append(line[0])
                                    vision_labels.append(f'V (f={force}, σ={sigma})')
                                    
                        except Exception as e:
                            print(f"Error processing folder {folder}: {e}")
                            continue
        
        ax.set_title(f'{metric} over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.grid(True)
    
    # Combine handles and labels, keeping methods separate
    all_handles = helbing_handles + vision_handles
    all_labels = helbing_labels + vision_labels
    
    # Create legend with 4 columns (2 for each method)
    n_items = len(all_handles)
    ncol = min(4, n_items)  # Use 4 columns or less if not enough items
    fig.legend(all_handles, all_labels,
              bbox_to_anchor=(1.05, 0.5),
              loc='center left',
              ncol=ncol,
              columnspacing=1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_pivot_table(df, method):
    """Helper function to create pivot table for heatmaps"""
    try:
        # For Test 1 (T and I values)
        if 'T=' in df['folder'].iloc[0] and 'I=' in df['folder'].iloc[0]:
            df['T'] = df['folder'].str.extract(r'T=(\d+)').astype(float)
            df['I'] = df['folder'].str.extract(r'I=(\d+)').astype(float)
            return df.pivot_table(values='value', 
                                index='T',
                                columns='I',
                                aggfunc='mean')
        
        # For Test 2 (Force and Sigma)
        elif 'force=' in df['folder'].iloc[0] and 'sigma=' in df['folder'].iloc[0]:
            df['force'] = df['folder'].str.extract(r'force=(\d+\.?\d*)').astype(float)
            df['sigma'] = df['folder'].str.extract(r'sigma=(\d+\.?\d*)').astype(float)
            return df.pivot_table(values='value', 
                                index='sigma',
                                columns='force',
                                aggfunc='mean')
        
        # For Tests 4 and 5 (Resolution and Agents)
        else:
            # Extract resolution and agent count, handling missing values
            resolution = df['folder'].str.extract(r'resolution=(\d+)')
            agents = df['folder'].str.extract(r'agents=(\d+)')
            
            # Convert to numeric, keeping NaN values
            df['resolution'] = pd.to_numeric(resolution[0], errors='coerce')
            df['agents'] = pd.to_numeric(agents[0], errors='coerce')
            
            # Drop rows with missing values
            df = df.dropna(subset=['resolution', 'agents'])
            
            # Create pivot table
            return df.pivot_table(values='value', 
                                index='agents',
                                columns='resolution',
                                aggfunc='mean')
    
    except Exception as e:
        print(f"Error creating pivot table for {method}: {e}")
        print("Sample folder name:", df['folder'].iloc[0] if not df.empty else "No data")
        return None

def calculate_method_difference(test_folder, metric, methods):
    """Helper function to calculate difference between methods"""
    data_by_method = {}
    for method in methods:
        data = []
        for folder in os.listdir(test_folder):
            if method in folder:
                result = load_data(os.path.join(test_folder, folder, "data.csv"))
                if result:
                    data_df, _ = result
                    final_value = data_df[metric].iloc[-1]
                    data.append((folder, final_value))
        
        if data:
            df = pd.DataFrame(data, columns=['folder', 'value'])
            pivot = create_pivot_table(df, method)
            data_by_method[method] = pivot
    
    return data_by_method[methods[0]] - data_by_method[methods[1]]

def create_specialized_time_series(data_folder, test_name, metric, methods, split_by=None, title=None):
    """Create time series plots split by a parameter
    
    Args:
        data_folder: Base data folder path
        test_name: Name of test folder
        metric: Metric to plot
        methods: List of methods to compare
        split_by: Parameter to split plots by (e.g., 'resolution', 'agents')
        title: Plot title
    """
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Collect all data first
    data_by_split = {}
    for method in methods:
        for folder in os.listdir(test_folder):
            if method not in folder:
                continue
                
            result = load_data(os.path.join(test_folder, folder, "data.csv"))
            if not result:
                continue
                
            data_df, _ = result
            
            # Extract split parameter value
            if split_by == 'resolution':
                value = int(folder.split('resolution=')[1].split('_')[0])
            elif split_by == 'agents':
                value = int(folder.split('agents=')[1].split('_')[0])
            else:
                value = 'all'
                
            key = (method, value)
            if key not in data_by_split:
                data_by_split[key] = []
            data_by_split[key].append(data_df[metric])
    
    # Get unique split values
    split_values = sorted(list(set(value for _, value in data_by_split.keys())))
    
    # Create subplots
    n_splits = len(split_values)
    fig, axes = plt.subplots(n_splits, 1, figsize=(12, 4*n_splits), sharex=True)
    if n_splits == 1:
        axes = [axes]
    
    # Store lines for legend
    legend_lines = []
    legend_labels = []
    
    # Plot each split
    for idx, split_value in enumerate(split_values):
        ax = axes[idx]
        
        for method in methods:
            key = (method, split_value)
            if key not in data_by_split:
                continue
            
            series_list = data_by_split[key]
            mean_series = pd.concat(series_list, axis=1).mean(axis=1)
            std_series = pd.concat(series_list, axis=1).std(axis=1)
            
            x = range(len(mean_series))
            line = ax.plot(x, mean_series, label=method)[0]
            ax.fill_between(x, 
                          mean_series - std_series,
                          mean_series + std_series,
                          alpha=0.2)
            
            if idx == 0:  # Only store legend items from first subplot
                legend_lines.append(line)
                legend_labels.append(method)
        
        if split_by:
            ax.set_title(f'{split_by}={split_value}')
        ax.grid(True)
        ax.set_ylabel(metric)
    
    # Add shared x-label
    axes[-1].set_xlabel('Step')
    
    # Add single legend outside plots
    fig.legend(legend_lines, legend_labels, 
              loc='center right', 
              bbox_to_anchor=(1.15, 0.5))
    
    # Add overall title
    if title:
        fig.suptitle(title, y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'{metric.lower()}_time_series.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_all_specialized_time_series(data_folder, test_name, methods, split_by=None):
    """Create specialized time series plots for all metrics"""
    # Efficiency over time
    create_specialized_time_series(
        data_folder, test_name, 'Efficiency', methods, split_by,
        title='Efficiency over Time'
    )
    
    # Civility over time
    create_specialized_time_series(
        data_folder, test_name, 'Civility', methods, split_by,
        title='Civility over Time'
    )
    
    # Computation time over time
    create_specialized_time_series(
        data_folder, test_name, 'TimeForInterval', methods, split_by,
        title='Computation Time over Time'
    )

def create_test1_heatmap(data_folder, test_name, metrics):
    """Create side-by-side heatmap for Test 1"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
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
        
        # Plot heatmap
        sns.heatmap(data, 
                   ax=axes[idx],
                   cmap=comfort_cmap if metric == 'Civility' else 'viridis',
                   xticklabels=range(1, 21),
                   yticklabels=range(20, 0, -1))  # Reverse T labels
        
        axes[idx].set_xlabel('I Value')
        axes[idx].set_ylabel('T Value')
        axes[idx].set_title(f'{metric}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '1_combined_metrics_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def find_most_similar_combinations(data_folder, test_name):
    """Find the most similar force/sigma combinations between Helbing and Vision methods"""
    test_folder = os.path.join(data_folder, test_name)
    
    # Store average metrics for each configuration
    helbing_metrics = {}
    vision_metrics = {}
    
    # Load data for both methods
    folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    for folder in folders:
        result = load_data(os.path.join(test_folder, folder, "data.csv"))
        if result:
            data_df, _ = result
            avg_efficiency = data_df['Efficiency'].mean()
            avg_civility = data_df['Civility'].mean()
            
            try:
                parts = folder.split('_')
                if 'Helbing_force=' in folder:
                    force = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('force='))
                    sigma = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('sigma='))
                    helbing_metrics[(force, sigma)] = (avg_efficiency, avg_civility)
                
                elif 'Vision_force=' in folder:
                    force = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('force='))
                    sigma = next(convert_float(part.split('=')[1]) for part in parts if part.startswith('sigma='))
                    vision_metrics[(force, sigma)] = (avg_efficiency, avg_civility)
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
                continue
    
    # Find most similar pair
    min_diff = float('inf')
    best_pair = None
    
    for h_params, h_metrics in helbing_metrics.items():
        for v_params, v_metrics in vision_metrics.items():
            diff = abs(h_metrics[0] - v_metrics[0]) + abs(h_metrics[1] - v_metrics[1])
            if diff < min_diff:
                min_diff = diff
                best_pair = (h_params, v_params)
    
    if best_pair is None:
        print("Warning: Could not find any matching parameter combinations")
        return (None, None)
        
    return best_pair

def plot_similar_combination_comparison(data_folder, test_name, helbing_params, vision_params):
    """Plot comparison of the most similar parameter combinations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    h_force, h_sigma = helbing_params
    v_force, v_sigma = vision_params
    
    # Create figure for time series comparison
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    metrics = ['Efficiency', 'Civility']
    
    # Plot time series for both methods
    for folder in os.listdir(test_folder):
        if (f'Helbing_force={h_force}_sigma={h_sigma}' in folder or 
            f'Vision_force={v_force}_sigma={v_sigma}' in folder):
            result = load_data(os.path.join(test_folder, folder, "data.csv"))
            if result:
                data_df, _ = result
                method = 'Helbing' if 'Helbing' in folder else 'Vision'
                color = 'blue' if method == 'Helbing' else 'red'
                
                for idx, metric in enumerate(metrics):
                    axes[idx].plot(data_df['Step'], data_df[metric], 
                                 label=f'{method} (force={h_force if method=="Helbing" else v_force}, sigma={h_sigma if method=="Helbing" else v_sigma})',
                                 color=color)
                    axes[idx].set_title(f'{metric} over Time')
                    axes[idx].set_xlabel('Step')
                    axes[idx].set_ylabel(metric)
                    axes[idx].grid(True)
                    axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '2_similar_combinations_time_series.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure for screenshot comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for idx, folder in enumerate(os.listdir(test_folder)):
        if (f'Helbing_force={h_force}_sigma={h_sigma}' in folder or 
            f'Vision_force={v_force}_sigma={v_sigma}' in folder):
            screenshot = plt.imread(os.path.join(test_folder, folder, "step_2000.png"))
            method = 'Helbing' if 'Helbing' in folder else 'Vision'
            idx = 0 if method == 'Helbing' else 1
            
            axes[idx].imshow(screenshot)
            axes[idx].set_title(f'{method} (force={h_force if method=="Helbing" else v_force}, sigma={h_sigma if method=="Helbing" else v_sigma})')
            axes[idx].axis('off')
    
    plt.savefig(os.path.join(results_folder, '2_similar_combinations_screenshots.png'),
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
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
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
            
            # Plot heatmap
            sns.heatmap(data, 
                       ax=ax,
                       cmap=comfort_cmap if metric == 'Civility' else 'viridis',
                       xticklabels=[f'{s:.1f}' for s in sigmas],
                       yticklabels=[f'{f:.1f}' for f in forces],
                       annot=True,
                       fmt='.1f',
                       **kwargs)
            
            ax.set_xlabel('Sigma (σ)')
            ax.set_ylabel('Force')
            ax.set_title(f'{method} {metric}')
            
            # Invert y-axis so lowest force is at bottom
            ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '2_parameter_heatmaps.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test2_screenshot_grid(data_folder, test_name, pattern_groups, title, filename):
    """Create screenshot grid specifically for Test 2"""
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
    plt.suptitle(title, y=0.95)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    plt.savefig(os.path.join(results_folder, filename), 
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
                               label=f"{combo['method']} (f={combo['force']}, σ={combo['sigma']})")
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over Time')
        ax.grid(True)
        ax.legend()
    
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
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"t={step*100}")
                
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
    
    print("Starting 1_screenshot_grid...")
    create_screenshot_grid(data_folder, test_name, None, "", "1_screenshot_grid.png")
    print("Completed 1_screenshot_grid")

def plot_test2(data_folder):
    """Plot results for Method Parameter Comparison"""
    test_name = "2_Force_and_Sigma_Values_to_match_paths"
    
    print("Starting 2_combined_time_series...")
    create_combined_time_series(data_folder, test_name,
                              metrics=['Efficiency', 'Civility'],
                              methods=['Helbing', 'Vision'],
                              filename='2_combined_time_series.png')
    print("Completed 2_combined_time_series")
    
    print("Starting 2_parameter_heatmap...")
    create_test2_heatmaps(data_folder, test_name,
                           metrics=['Efficiency', 'Civility'],
                           methods=['Helbing', 'Vision'])
    print("Completed 2_parameter_heatmap")
    
    print("Starting 2_parameter_grid...")
    create_test2_screenshot_grid(data_folder, test_name, None,
                               "Trail Formation with Different Force and Sigma Values",
                               filename='2_parameter_grid.png')
    print("Completed 2_parameter_grid")
    
    print("Starting 2_similar_combinations_time_series...")
    create_test2_similar_combinations_time_series(data_folder, test_name)
    print("Completed 2_similar_combinations_time_series")
    
    print("Starting 2_similar_combinations_screenshots...")
    create_test2_similar_combinations_screenshots(data_folder, test_name)
    print("Completed 2_similar_combinations_screenshots")

def plot_test3(data_folder):
    """Plot results for Vision Method Sampling Analysis"""
    test_name = "3_Sample_points_paired_with_forces_and_sigmas_small"
    
    # ... similar structure to test2 but for sampling configurations ...

def plot_test4(data_folder):
    """Plot results for Performance Scaling Analysis"""
    test_name = "4_Performance_with_varying_agent_count_and_Resolution"
    
    # Create standard plots
    create_combined_time_series(data_folder, test_name,
                              metrics=['Efficiency', 'Civility'],
                              methods=['Helbing', 'Vision'])
    
    create_combined_heatmap(data_folder, test_name,
                           metrics=['Efficiency', 'Civility'],
                           methods=['Helbing', 'Vision'])
    
    create_performance_comparison(data_folder, test_name,
                                methods=['Helbing', 'Vision'])
    
    # Create specialized time series plots split by resolution
    create_all_specialized_time_series(data_folder, test_name,
                                     methods=['Helbing', 'Vision'],
                                     split_by='resolution')
    
    # Create specialized time series plots split by agent count
    create_all_specialized_time_series(data_folder, test_name,
                                     methods=['Helbing', 'Vision'],
                                     split_by='agents')

def plot_test5(data_folder):
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
    
    print("Starting 5_performance_bars...")
    create_performance_bar_graph(data_folder, test_name,
                               methods=methods,
                               method_labels=method_labels,
                               filename='5_performance_bars.png')
    print("Completed 5_performance_bars")
    
    print("Starting 5_performance_by_resolution...")
    create_performance_by_resolution(data_folder, test_name,
                                   methods=methods,
                                   method_labels=method_labels)
    print("Completed 5_performance_by_resolution")
    
    print("Starting 5_performance_bars_700...")
    create_performance_bar_graph_700(data_folder, test_name,
                                   methods=methods,
                                   method_labels=method_labels)
    print("Completed 5_performance_bars_700")
    
    print("Starting 5_time_series_by_resolution...")
    create_test5_time_series_by_resolution(data_folder, test_name,
                                         methods=methods,
                                         method_labels=method_labels)
    print("Completed 5_time_series_by_resolution")
    
    print("Starting 5_screenshot_grid...")
    #create_test5_screenshot_grid(data_folder, test_name,
    #                            methods=methods,
    #                            method_labels=method_labels)
    #print("Completed 5_screenshot_grid")
    
    print("Starting 5_screenshot_grid_full...")
    #create_test5_full_screenshot_grid(data_folder, test_name,
    #                                methods=methods,
    #                                method_labels=method_labels)
    #print("Completed 5_screenshot_grid_full")

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
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
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
    
    ax.set_xlabel('Resolution-Agent Count Combinations')
    ax.set_ylabel('Average Time per Step (ms)')
    ax.set_title('Performance by Resolution and Agent Count')
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels([f'R{res}-A{agents}' for res, agents in combinations], rotation=45)
    ax.yaxis.set_major_locator(MultipleLocator(20))  # Ensure 20ms is shown on y-axis
    ax.legend()
    
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
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
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
    
    ax.set_xlabel('Agent Count')
    ax.set_ylabel('Average Time per Step (ms)')
    ax.set_title('Performance at Resolution 700')
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(agent_counts)
    ax.yaxis.set_major_locator(MultipleLocator(20))  # Ensure 20ms is shown on y-axis
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '5_performance_bars_700.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test5_time_series(data_folder, test_name, methods, method_labels):
    """Create time series plot showing all configurations"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    agent_counts = sorted(set(int(f.split('agents=')[1].split('_')[0]) 
                         for f in all_folders if 'agents=' in f))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create different line styles and more distinct colors
    linestyles = ['-', '--', ':', '-.', '-']
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for method_idx, (method, method_label) in enumerate(zip(methods, method_labels)):
        for res in resolutions:
            for agents in agent_counts:
                pattern = f"{method}_resolution={res}_agents={agents}"
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
                    avg_times = np.mean(times, axis=0)
                    steps = np.arange(100, (len(avg_times) + 1) * 100, 100)
                    label = f"{method_label} (R{res}-A{agents})"
                    ax.plot(steps, avg_times, 
                           label=label,
                           linestyle=linestyles[method_idx],
                           color=colors[method_idx],
                           alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Computation Time per 100 Steps')
    
    # Create legend with 4 columns
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=4)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '5_time_series.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test5_time_series_by_resolution(data_folder, test_name, methods, method_labels):
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
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
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
                           alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Time per Step (ms)')
        ax.set_title(f'Computation Time per Step (Resolution {resolution})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        ax.yaxis.set_major_locator(MultipleLocator(100))  # Set y-axis ticks every 100ms
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'5_time_series_res_{resolution}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_performance_by_resolution(data_folder, test_name, methods, method_labels):
    """Create bar graph showing performance by resolution"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    
    all_folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
    resolutions = sorted(set(int(f.split('resolution=')[1].split('_')[0]) 
                        for f in all_folders if 'resolution=' in f))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
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
    
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Average Time per Step (ms)')  # Updated label
    ax.set_title('Performance by Resolution')
    ax.yaxis.set_major_locator(MultipleLocator(20))  # Ensure 20ms is shown on y-axis
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '5_performance_by_resolution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test5_screenshot_grid(data_folder, test_name, methods, method_labels):
    """Create grid of final screenshots for each configuration"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create a 1x5 grid (methods side by side)
    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1])
    
    for method_idx, (method, label) in enumerate(zip(methods, method_labels)):
        # Find a folder with resolution=300 and agents=30 for this method
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
                ax.text(0.5, 0.5, "No screenshot found", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No matching folder found", ha='center', va='center')
        
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Final Trail Formation (Resolution 100, 30 Agents)', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '5_screenshot_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_test5_full_screenshot_grid(data_folder, test_name, methods, method_labels):
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
    fig = plt.figure(figsize=(25, 8))
    outer_grid = gridspec.GridSpec(len(agent_counts), len(resolutions))
    
    for i, agents in enumerate(agent_counts):
        for j, res in enumerate(resolutions):
            # Create inner grid for methods
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, len(methods), 
                subplot_spec=outer_grid[i, j], wspace=0.1, hspace=0.1)
            
            for method_idx, (method, label) in enumerate(zip(methods, method_labels)):
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
                        ax.text(0.5, 0.5, "No screenshot", ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                
                # Add labels
                if i == 0 and method_idx == len(methods)//2:  # Top row, middle method
                    ax.set_title(f'Resolution {res}')
                if j == 0 and method_idx == 0:  # Left column, first method
                    ax.set_ylabel(f'{agents} Agents')

                
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.suptitle('Trail Formation Grid\nDisplay order per group: Helbing, Vision (30 pts), Vision (75 pts), Vision (200 pts), Vision (1000 pts)', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, '5_screenshot_grid_full.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_folder = "../ExperimentData"
    if not os.path.exists(data_folder):
        print(f"Creating data folder: {os.path.abspath(data_folder)}")
        os.makedirs(data_folder)
    
    #plot_test1(data_folder)
    #plot_test2(data_folder)
    #plot_test3(data_folder)
    #plot_test4(data_folder)
    plot_test5(data_folder)