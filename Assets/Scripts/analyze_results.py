import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Fix for colormap
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import itertools

def convert_float(value):
    """Convert comma-formatted number to float"""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def load_data(filename):
    """Load data from CSV file and return DataFrame and config lines"""
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
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(30, 16*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    # Create color gradients
    T_colors = plt.cm.RdYlGn(np.linspace(0, 1, 20))  # Red to Green for T values
    I_colors = plt.cm.Blues(np.linspace(0.3, 1, 20))  # Blues for I values
    
    # Store line styles for consistent plotting
    line_styles = {}
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_count = 0
        
        # Plot a line for each T/I combination
        # Reverse the order of T to make T=1 at the top
        for T in range(20, 0, -1):  # Changed to count down from 20 to 1
            for I in range(1, 21):
                expected_index = ((T-1) * 20) + I
                expected_folder = f"Helbing_T={T}_I={I}_repetition=1_index={expected_index}"
                
                if (T, I) in data_paths and os.path.exists(data_paths[(T, I)]):
                    result = load_data(data_paths[(T, I)])
                    if result:
                        data_df, _ = result
                        if len(data_df) >= 20:
                            steps = range(100, 2100, 100)
                            label = f'T={T},I={I}' if metric_idx == 0 else None
                            
                            if (T,I) not in line_styles:
                                # Blend T and I colors
                                T_color = T_colors[T-1]
                                I_color = I_colors[I-1]
                                blended_color = (T_color + I_color) / 2
                                line_styles[(T,I)] = {'color': blended_color, 'alpha': 0.5}
                            
                            ax.plot(steps, data_df[metric], 
                                  label=label, **line_styles[(T,I)])
                            plotted_count += 1
                        else:
                            print(f"✗ Not enough data points in: {expected_folder} (only {len(data_df)} points)")
                else:
                    print(f"✗ Missing folder: {expected_folder}")
        
        print(f"\nPlotted {plotted_count} lines for {metric} out of expected 400")
        
        ax.set_title(f'{metric} over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.grid(True)
        
        # Crop the plot by adjusting the axis limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x_range = xmax - xmin
        y_range = ymax - ymin
        
        # Apply 25% crop on left and right, 10% on top and bottom
        ax.set_xlim(xmin + x_range * 0.25, xmax - x_range * 0.25)
        ax.set_ylim(ymin + y_range * 0.10, ymax - y_range * 0.10)
        
        if metric_idx == 0 and plotted_count > 0:  # Only add legend to first plot
            # Move legend to top of plot
            ax.legend(bbox_to_anchor=(0.5, 1.4), loc='center', 
                     ncol=4, borderaxespad=0.)
    
    # Adjust figure size and add more space at top for legend
    plt.gcf().set_size_inches(30, 28*len(metrics))  # Made taller to accommodate top legend
    plt.subplots_adjust(top=0.85)  # Leave more space at top for legend
    plt.savefig(os.path.join(results_folder, '1_metrics_time_series.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_screenshot_grid(data_folder, test_name, pattern_groups, title, filename):
    """Create a grid of screenshots from simulation runs"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # For Test 1, create a grid based on T and I values
    if "1_I_and_T" in test_name:
        fig, axes = plt.subplots(20, 20, figsize=(40, 40))  # 20x20 grid for T and I values
        
        # Get list of folders excluding .meta files
        folders = [f for f in os.listdir(test_folder) if not f.endswith('.meta')]
        
        for T in range(1, 21):
            for I in range(1, 21):
                ax = axes[T-1, I-1]  # T=1 at top (index 0), T=20 at bottom (index 19)
                
                # Calculate expected index
                expected_index = ((T-1) * 20) + I
                
                # Find the step_2000.png for this T,I combination
                pattern = f"Helbing_T={T}_I={I}_repetition=1_index={expected_index}"
                screenshot_path = None
                
                for folder in folders:
                    if pattern in folder:
                        potential_path = os.path.join(test_folder, folder, "step_2000.png")
                        if os.path.exists(potential_path):
                            screenshot_path = potential_path
                            break
                
                if screenshot_path:
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
                    ax.text(0.5, 0.5, "No image", ha='center', va='center')
                
                # Show I values on top row, T values on left column
                if T == 1:  # Top row
                    ax.set_title(f'I={I}')
                if I == 1:   # Left column
                    ax.set_ylabel(f'T={T}')
                
                ax.set_xticks([])
                ax.set_yticks([])
    
    else:
        # Original implementation for other tests...
        n_groups = len(pattern_groups)
        n_cols = min(5, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols)
        
        for idx, (group_name, pattern) in enumerate(pattern_groups.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Find matching screenshot
            found = False
            for folder in os.listdir(test_folder):
                if pattern.replace('*', '') in folder:
                    screenshot_path = os.path.join(test_folder, folder, "final_state.png")
                    if os.path.exists(screenshot_path):
                        img = plt.imread(screenshot_path)
                        ax.imshow(img)
                        found = True
                        break
            
            if not found:
                ax.text(0.5, 0.5, "No screenshot", ha='center', va='center')
            
            ax.set_title(group_name)
            ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), dpi=300, bbox_inches='tight')
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

def create_combined_time_series(data_folder, test_name, metrics, methods, filename):
    """Create combined time series plots"""
    test_folder = os.path.join(data_folder, test_name)
    results_folder = os.path.join(test_folder, "_Results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create subplot for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for method in methods:
            # Collect all time series for this method
            all_series = []
            for folder in os.listdir(test_folder):
                if method in folder:
                    result = load_data(os.path.join(test_folder, folder, "data.csv"))
                    if result:
                        data_df, _ = result
                        all_series.append(data_df[metric])
            
            if all_series:
                # Calculate mean and std
                mean_series = pd.concat(all_series, axis=1).mean(axis=1)
                std_series = pd.concat(all_series, axis=1).std(axis=1)
                
                # Plot with confidence interval
                x = range(len(mean_series))
                ax.plot(x, mean_series, label=method)
                ax.fill_between(x, 
                              mean_series - std_series,
                              mean_series + std_series,
                              alpha=0.2)
        
        ax.set_title(f'{metric} over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
    
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
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
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

def plot_test1(data_folder):
    """Plot results for Trail Formation Parameter Analysis"""
    test_name = "1_I_and_T_values_for_different_trails"
    
    # Create specialized heatmap for Test 1
    create_test1_heatmap(data_folder, test_name, ['Efficiency', 'Civility'])
    
    # Create time series with all T/I combinations
    create_test1_time_series(data_folder, test_name, ['Efficiency', 'Civility'])
    
    # Create T/I parameter grid screenshots
    create_screenshot_grid(data_folder, test_name, None,
                         "Trail Formation with Different T and I Values",
                         filename='1_screenshot_grid.png')

def plot_test2(data_folder):
    """Plot results for Method Parameter Comparison"""
    test_name = "2_Force_and_Sigma_Values_to_match_paths"
    
    # Create standard plots for both methods
    create_combined_time_series(data_folder, test_name,
                              metrics=['Efficiency', 'Civility'],
                              methods=['Helbing', 'Vision'])
    
    create_combined_heatmap(data_folder, test_name,
                           metrics=['Efficiency', 'Civility'],
                           methods=['Helbing', 'Vision'])
    
    create_performance_comparison(data_folder, test_name,
                                methods=['Helbing', 'Vision'])
    
    # Create force/sigma parameter grid screenshots
    pattern_groups = {
        'Helbing': 'Helbing_force=*_sigma=*',
        'Vision': 'Vision_force=*_sigma=*'
    }
    create_screenshot_grid(data_folder, test_name, pattern_groups,
                         "Trail Formation with Different Force and Sigma Values")

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
    
    # Standard plots
    create_combined_time_series(data_folder, test_name,
                              metrics=['Efficiency', 'Civility'],
                              methods=['Helbing', 'Vision'])
    
    create_combined_heatmap(data_folder, test_name,
                           metrics=['Efficiency', 'Civility'],
                           methods=['Helbing', 'Vision'])
    
    create_performance_comparison(data_folder, test_name,
                                methods=['Helbing', 'Vision'])
    
    # Special comparison of different sampling configurations
    sampling_configs = [
        "Vision_arcs=3_first=5_last=5",   # Minimal
        "Vision_arcs=5_first=10_last=20",  # Medium
        "Vision_arcs=10_first=20_last=20", # High
        "Vision_arcs=20_first=50_last=50"  # Very High
    ]
    
    # Create screenshot grid of resolution=100, agents=30, rep=1
    pattern_groups = {
        'Helbing': 'Helbing_resolution=100_agents=30_repetition=1',
        **{f'Vision_{i}': f'{config}_resolution=100_agents=30_repetition=1'
           for i, config in enumerate(sampling_configs)}
    }
    create_screenshot_grid(data_folder, test_name, pattern_groups,
                         "Trail Formation with Different Sampling Configurations",
                         filename='5_screenshot_grid.png')

if __name__ == "__main__":
    data_folder = "../ExperimentData"
    if not os.path.exists(data_folder):
        print(f"Creating data folder: {os.path.abspath(data_folder)}")
        os.makedirs(data_folder)
    
    plot_test1(data_folder)
    #plot_test2(data_folder)
    #plot_test3(data_folder)
    #plot_test4(data_folder)
    #plot_test5(data_folder)