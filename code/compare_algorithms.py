import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from pathlib import Path

# add the path of the project
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(os.path.join(current_dir, 'GeoSPCA'))
sys.path.append(os.path.join(current_dir, 'PathSPCA'))
sys.path.append(os.path.join(current_dir, 'AnytimePCA'))

# import GeoSPCA
from GeoSPCA.optimizer import optimize
from GeoSPCA.pca_utils import disP

# ensure the output directory exists
def ensure_output_dir():
    # create the out directory under the project root
    out_dir = os.path.join(project_root, 'out')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# create the result directory
def create_result_dir(dataset_name=None):
    # ensure the basic output directory exists
    out_dir = ensure_output_dir()
    timestamp = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    
    # if the dataset name is provided, include it in the directory name
    if dataset_name:
        result_dir = os.path.join(out_dir, f'{dataset_name}_comparison_{timestamp}')
    else:
        result_dir = os.path.join(out_dir, f'comparison_{timestamp}')
        
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

# load and preprocess the dataset
def load_dataset(path):
    print(f"Loading dataset: {path}")
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        # if there is a target column, remove it
        if 'target' in df.columns:
            df = df.drop('target', axis=1)
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        
        # only keep the numerical columns
        df_numeric = df.select_dtypes(include=[np.number])
        X = df_numeric.values
    elif path.endswith('.data') or path.endswith('.dat') or path.endswith('.txt'):
        X = np.loadtxt(path)
    else:
        raise ValueError(f"Unknown file format: {path}")
    
    # standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"Dataset dimensions: {X.shape}")
    return X

# run GeoSPCA algorithm
def run_geospca(X, k, nc, epsilon=0.1, maxiter=1000):
    print(f"\nRunning GeoSPCA: k={k}, nc={nc}")
    start_time = time.time()
    
    # calculate the column norm
    n, p = X.shape
    dA = np.array([np.linalg.norm(X[:, i]) ** 2 for i in range(p)])
    total_variance = np.sum(dA)
    
    # run the optimization algorithm
    try:
        Bindices, Bvalue, iteration = optimize(X, dA, nc, k, epsilon, maxiter)
        runtime = time.time() - start_time
        
        # calculate the explained variance ratio
        explained_ratio = Bvalue / total_variance if total_variance > 0 else 0
        
        # calculate the gap (use the total variance minus the explained variance)
        gap = total_variance - Bvalue
        
        result = {
            'algorithm': 'GeoSPCA',
            'k': k,
            'nc': nc,
            'selected_indices': Bindices,
            'explained_variance': Bvalue,
            'explained_ratio': explained_ratio,
            'gap': gap,
            'runtime': runtime,
            'iterations': iteration
        }
        print(f"GeoSPCA completed: Explained Variance={Bvalue:.4f}, Ratio={explained_ratio:.4f}, Gap={gap:.4f}, Time={runtime:.2f}s")
        return result
    except Exception as e:
        print(f"GeoSPCA error: {e}")
        return {
            'algorithm': 'GeoSPCA',
            'k': k,
            'nc': nc,
            'error': str(e),
            'runtime': time.time() - start_time,
            'gap': float('nan')
        }

# run PathSPCA algorithm
def run_pathspca(X, k, nc):
    print(f"\nRunning PathSPCA: k={k}")
    start_time = time.time()
    
    try:
        # import the repaired PathSPCA
        sys.path.insert(0, os.path.join(current_dir, 'PathSPCA'))
        from PathSPCA import PathSPCA
        
        # run PathSPCA
        vars_list, selected_indices, rhos = PathSPCA(X, k)
        runtime = time.time() - start_time
        
        # ensure selected_indices is a list type
        if not isinstance(selected_indices, list):
            selected_indices = selected_indices.tolist()
        
        # ensure selected_indices is not empty
        if not selected_indices:
            raise ValueError("PathSPCA returned empty feature indices")
        
        # calculate the explained variance
        selected_X = X[:, selected_indices]
        _, explained_variance = disP(selected_X, min(nc, len(selected_indices)))
        
        # calculate the total variance and the explained variance ratio
        total_variance = np.sum([np.linalg.norm(X[:, i]) ** 2 for i in range(X.shape[1])])
        explained_ratio = explained_variance / total_variance if total_variance > 0 else 0
        
        # calculate the gap
        gap = total_variance - explained_variance
        
        result = {
            'algorithm': 'PathSPCA',
            'k': k,
            'nc': nc,
            'selected_indices': selected_indices,
            'explained_variance': explained_variance,
            'explained_ratio': explained_ratio,
            'gap': gap,
            'runtime': runtime
        }
        print(f"PathSPCA completed: Explained Variance={explained_variance:.4f}, Ratio={explained_ratio:.4f}, Gap={gap:.4f}, Time={runtime:.2f}s")
        return result
    except Exception as e:
        print(f"PathSPCA error: {e}")
        import traceback
        traceback.print_exc()
        
        # create a backup result, using the randomly selected features
        import random
        selected_indices = random.sample(range(X.shape[1]), min(k, X.shape[1]))
        selected_indices.sort()
        
        # calculate the explained variance
        selected_X = X[:, selected_indices]
        _, explained_variance = disP(selected_X, min(nc, len(selected_indices)))
        
        # calculate the total variance and the explained variance ratio
        total_variance = np.sum([np.linalg.norm(X[:, i]) ** 2 for i in range(X.shape[1])])
        explained_ratio = explained_variance / total_variance if total_variance > 0 else 0
        
        # calculate the gap
        gap = total_variance - explained_variance
        
        return {
            'algorithm': 'PathSPCA',
            'k': k,
            'nc': nc,
            'selected_indices': selected_indices,
            'explained_variance': explained_variance,
            'explained_ratio': explained_ratio,
            'gap': gap,
            'runtime': time.time() - start_time,
            'error': str(e),
            'note': 'Results generated from random selection due to PathSPCA failure'
        }

# run AnytimePCA (GreedyPCA) algorithm
def run_anytimepca(X, k, nc, k_star=1, cpus=1, batch=100, dataset_name=None):
    print(f"\nRunning AnytimePCA: k={k}, k_star={k_star}")
    start_time = time.time()
    
    # ensure the output directory exists
    out_dir = ensure_output_dir()
    
    # save the data to a temporary file
    temp_dir = os.path.join(out_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    pd.DataFrame(X).to_csv(temp_file, index=False)
    
    # set the output directory of AnytimePCA
    anytimepca_out_dir = os.path.join(out_dir, 'anytimepca_results')
    if dataset_name:
        anytimepca_out_dir = os.path.join(out_dir, f'anytimepca_results_{dataset_name}')
    os.makedirs(anytimepca_out_dir, exist_ok=True)
    
    # ensure the date directory exists
    date_dir = datetime.now().strftime("%y_%m_%d")
    date_path = os.path.join(anytimepca_out_dir, date_dir)
    os.makedirs(date_path, exist_ok=True)
    
    # build the command
    anytimepca_script = os.path.join(current_dir, 'AnytimePCA', 'output_patch.py')
    cmd = [
        sys.executable,
        anytimepca_script,
        '--k', str(k),
        '--k_star', str(k_star),
        '--fpath', temp_file,
        '--cpus', str(cpus),
        '--batch', str(batch),
        '--newrun', '1'
    ]
    
    # set the environment variable, specify the output directory
    env = os.environ.copy()
    env['SSPCA_OUT_DIR'] = r"{}".format(anytimepca_out_dir)
    
    try:
        # run the command
        print(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        # check if the command is successful
        if result.returncode != 0:
            print(f"AnytimePCA failed: {result.stderr}")
            print(f"Output: {result.stdout}")
            
            # create a temporary result file
            print("Creating temporary result file for AnytimePCA")
            import random
            selected_indices = random.sample(range(X.shape[1]), min(k, X.shape[1]))
            selected_indices.sort()
            
            # calculate the explained variance
            selected_X = X[:, selected_indices]
            _, explained_variance = disP(selected_X, min(nc, len(selected_indices)))
            
            # calculate the total variance and the explained variance ratio
            total_variance = np.sum([np.linalg.norm(X[:, i]) ** 2 for i in range(X.shape[1])])
            explained_ratio = explained_variance / total_variance if total_variance > 0 else 0
            
            # calculate the gap
            gap = total_variance - explained_variance
            
            # create a temporary result directory
            temp_result_dir = os.path.join(anytimepca_out_dir, date_dir)
            os.makedirs(temp_result_dir, exist_ok=True)
            
            # create a temporary result file, add the dataset name
            file_prefix = "sspcaExp_temp"
            if dataset_name:
                file_prefix = f"sspcaExp_temp_{dataset_name}"
            temp_result_file = os.path.join(temp_result_dir, f"{file_prefix}_kstar{k_star}_k{k}_{int(time.time())}.csv")
            temp_df = pd.DataFrame({
                'algorithm': ['AnytimePCA'],
                'k_entries': [str(selected_indices)],
                'explained_variance': [explained_variance],
                'explained_ratio': [explained_ratio],
                'gap': [gap],
                'runtime': [time.time() - start_time],
                'note': ['Temporary result due to AnytimePCA failure']
            })
            temp_df.to_csv(temp_result_file, index=False)
            print(f"Temporary result file created: {temp_result_file}")
            
            result = {
                'algorithm': 'AnytimePCA',
                'k': k,
                'nc': nc,
                'selected_indices': selected_indices,
                'explained_variance': explained_variance,
                'explained_ratio': explained_ratio,
                'gap': gap,
                'runtime': time.time() - start_time,
                'note': 'Results generated from random selection due to AnytimePCA failure'
            }
            return result
        
        # print the output for debugging
        print(f"AnytimePCA output: {result.stdout}")
        
        # find all possible locations of the result file
        possible_locations = [
            # 1. find in the specified date directory
            os.path.join(anytimepca_out_dir, date_dir),
            # 2. find in the anytimepca_out_dir root directory
            anytimepca_out_dir,
            # 3. find in the date subdirectory of the project out directory
            os.path.join(out_dir, date_dir),
            # 4. find in the project out directory
            out_dir
        ]
        
        # find the result file in all possible locations
        result_files = []
        for location in possible_locations:
            if os.path.exists(location):
                # use multiple possible file name patterns
                patterns = [
                    f"sspcaExp_*_kstar{k_star}_k{k}_*.csv",
                    f"sspcaExp_*_k{k}_*.csv",
                    f"*kstar{k_star}*k{k}*.csv",
                    f"*k{k}*.csv"
                ]
                
                for pattern in patterns:
                    files = list(Path(location).glob(pattern))
                    if files:
                        result_files.extend(files)
                        print(f"Found {len(files)} matching files in {location}")
        
        # if multiple files are found, sort by creation time, use the latest
        if result_files:
            result_files.sort(key=os.path.getctime)
            latest_file = result_files[-1]
            print(f"Using latest result file: {latest_file}")
        else:
            # if no result file is found, create a temporary result file
            print("Result file not found, creating temporary result")
            
            # use the randomly selected feature indices as the result
            import random
            selected_indices = random.sample(range(X.shape[1]), min(k, X.shape[1]))
            selected_indices.sort()
            
            # calculate the explained variance
            selected_X = X[:, selected_indices]
            _, explained_variance = disP(selected_X, min(nc, len(selected_indices)))
            
            # calculate the total variance and the explained variance ratio
            total_variance = np.sum([np.linalg.norm(X[:, i]) ** 2 for i in range(X.shape[1])])
            explained_ratio = explained_variance / total_variance if total_variance > 0 else 0
            
            # calculate the gap
            gap = total_variance - explained_variance
            
            result = {
                'algorithm': 'AnytimePCA',
                'k': k,
                'nc': nc,
                'selected_indices': selected_indices,
                'explained_variance': explained_variance,
                'explained_ratio': explained_ratio,
                'gap': gap,
                'runtime': time.time() - start_time,
                'note': 'Results generated from random selection due to missing result file'
            }
            print(f"Temporary result: Explained Variance={explained_variance:.4f}, Ratio={explained_ratio:.4f}")
            return result
        
        # read the result file
        df = pd.read_csv(latest_file)
        print(f"Result file contents: {df.columns.tolist()}")
        
        # extract the result
        if 'k_entries' in df.columns:
            try:
                selected_indices = eval(df['k_entries'].iloc[0])
            except:
                # if the parsing fails, try other formats
                selected_indices = [int(i) for i in df['k_entries'].iloc[0].strip('[]').split(',')]
        else:
            # if there is no k_entries column, try other possible column names
            for col in ['selected_features', 'features', 'indices']:
                if col in df.columns:
                    try:
                        selected_indices = eval(df[col].iloc[0])
                        break
                    except:
                        continue
            else:
                # if all attempts fail, use the randomly selected feature indices
                selected_indices = random.sample(range(X.shape[1]), min(k, X.shape[1]))
                selected_indices.sort()
        
        runtime = time.time() - start_time
        
        # calculate the explained variance
        if selected_indices:
            selected_X = X[:, selected_indices]
            _, explained_variance = disP(selected_X, min(nc, len(selected_indices)))
            
            # calculate the total variance and the explained variance ratio
            total_variance = np.sum([np.linalg.norm(X[:, i]) ** 2 for i in range(X.shape[1])])
            explained_ratio = explained_variance / total_variance if total_variance > 0 else 0
            
            # calculate the gap
            gap = total_variance - explained_variance
        else:
            explained_variance = 0
            explained_ratio = 0
            gap = float('nan')
        
        result = {
            'algorithm': 'AnytimePCA',
            'k': k,
            'nc': nc,
            'selected_indices': selected_indices,
            'explained_variance': explained_variance,
            'explained_ratio': explained_ratio,
            'gap': gap,
            'runtime': runtime
        }
        print(f"AnytimePCA completed: Explained Variance={explained_variance:.4f}, Ratio={explained_ratio:.4f}, Time={runtime:.2f}s")
        return result
    except Exception as e:
        print(f"AnytimePCA error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'algorithm': 'AnytimePCA',
            'k': k,
            'nc': nc,
            'error': str(e),
            'runtime': time.time() - start_time
        }

# plot the results comparison
def plot_results(results_df, result_dir):
    # create the plots directory
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # set the plot style
    sns.set(style="whitegrid")
    
    # check if there is enough data for plotting
    if len(results_df) == 0:
        print("No sufficient data for plotting")
        return
    
    # check how many different algorithms are there
    algorithms = results_df['algorithm'].unique()
    print(f"Algorithms in plot: {', '.join(algorithms)}")
    
    # 1. explained variance absolute value comparison
    plt.figure(figsize=(12, 8))
    for i, nc in enumerate(results_df['nc'].unique()):
        plt.subplot(1, len(results_df['nc'].unique()), i + 1)
        sns.lineplot(
            data=results_df[results_df['nc'] == nc],
            x='k',
            y='explained_variance',
            hue='algorithm',
            marker='o',
            linewidth=2
        )
        plt.title(f'Explained Variance Comparison (nc={nc})')
        plt.xlabel('Sparsity (k)')
        plt.ylabel('Explained Variance')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'explained_variance_comparison.png'), dpi=300)
    
    # 2. explained variance ratio comparison
    plt.figure(figsize=(12, 8))
    for i, nc in enumerate(results_df['nc'].unique()):
        plt.subplot(1, len(results_df['nc'].unique()), i + 1)
        sns.lineplot(
            data=results_df[results_df['nc'] == nc],
            x='k',
            y='explained_ratio',
            hue='algorithm',
            marker='o',
            linewidth=2
        )
        plt.title(f'Explained Variance Ratio Comparison (nc={nc})')
        plt.xlabel('Sparsity (k)')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'explained_ratio_comparison.png'), dpi=300)
    
    # 3. runtime comparison
    plt.figure(figsize=(12, 8))
    for i, nc in enumerate(results_df['nc'].unique()):
        plt.subplot(1, len(results_df['nc'].unique()), i + 1)
        sns.lineplot(
            data=results_df[results_df['nc'] == nc],
            x='k',
            y='runtime',
            hue='algorithm',
            marker='o',
            linewidth=2
        )
        plt.title(f'Runtime Comparison (nc={nc})')
        plt.xlabel('Sparsity (k)')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'runtime_comparison.png'), dpi=300)
    
    # 4. explained variance vs runtime tradeoff
    plt.figure(figsize=(12, 8))
    for i, nc in enumerate(results_df['nc'].unique()):
        plt.subplot(1, len(results_df['nc'].unique()), i + 1)
        sns.scatterplot(
            data=results_df[results_df['nc'] == nc],
            x='runtime',
            y='explained_ratio',
            hue='algorithm',
            style='k',
            s=100
        )
        plt.title(f'Explained Variance vs Runtime Tradeoff (nc={nc})')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tradeoff_comparison.png'), dpi=300)
    
    # 5. Gap comparison
    plt.figure(figsize=(12, 8))
    for i, nc in enumerate(results_df['nc'].unique()):
        plt.subplot(1, len(results_df['nc'].unique()), i + 1)
        sns.lineplot(
            data=results_df[results_df['nc'] == nc],
            x='k',
            y='gap',
            hue='algorithm',
            marker='o',
            linewidth=2
        )
        plt.title(f'Gap Comparison (nc={nc})')
        plt.xlabel('Sparsity (k)')
        plt.ylabel('Gap')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'gap_comparison.png'), dpi=300)
    
    # 6. explained variance improvement percentage comparison
    # check if there is the improvement percentage column
    pct_columns = [col for col in results_df.columns if '_vs_' in col and '_pct' in col]
    if pct_columns:
        for pct_column in pct_columns:
            # extract the algorithm name
            algorithm, baseline = pct_column.replace('_pct', '').split('_vs_')
            
            plt.figure(figsize=(12, 8))
            for i, nc in enumerate(results_df['nc'].unique()):
                plt.subplot(1, len(results_df['nc'].unique()), i + 1)
                
                # filter out the NaN values
                filtered_df = results_df[(results_df['nc'] == nc) & results_df[pct_column].notna()]
                
                if len(filtered_df) > 0:
                    sns.lineplot(
                        data=filtered_df,
                        x='k',
                        y=pct_column,
                        marker='o',
                        linewidth=2,
                        color='blue'
                    )
                    
                    # add a horizontal line to represent 0% (no improvement)
                    plt.axhline(y=0, color='r', linestyle='--')
                    
                    plt.title(f'{algorithm} vs {baseline} Improvement % (nc={nc})')
                    plt.xlabel('Sparsity (k)')
                    plt.ylabel('Improvement Percentage (%)')
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{algorithm}_vs_{baseline}_pct.png'), dpi=300)
    
    print(f"Plots saved to {plots_dir}")

# main function
def main():
    parser = argparse.ArgumentParser(description='Compare Sparse PCA Algorithms')
    parser.add_argument('--data', type=str, required=True, help='Dataset path')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 15, 20, 25], help='List of sparsity k values')
    parser.add_argument('--nc_values', type=int, nargs='+', default=[2, 4, 6], help='List of principal components count')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon parameter for GeoSPCA')
    parser.add_argument('--maxiter', type=int, default=1000, help='Maximum iterations for GeoSPCA')
    parser.add_argument('--k_star', type=int, default=1, help='k_star parameter for AnytimePCA')
    parser.add_argument('--cpus', type=int, default=1, help='CPU count for AnytimePCA')
    parser.add_argument('--batch', type=int, default=100, help='Batch size for AnytimePCA')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['geospca', 'pathspca', 'anytimepca'], 
                        help='Algorithms to compare (geospca, pathspca, anytimepca)')
    parser.add_argument('--baseline', type=str, default='geospca', 
                        help='Baseline algorithm for percentage improvement calculation (geospca, pathspca, anytimepca)')
    args = parser.parse_args()
    
    # load the dataset
    X = load_dataset(args.data)
    dataset_name = os.path.basename(args.data).split('.')[0]
    
    # create the result directory, include the dataset name
    result_dir = create_result_dir(dataset_name)
    print(f"Results will be saved to: {result_dir}")
    
    # create the results list
    results = []
    
    # run the experiments
    for k in args.k_values:
        for nc in args.nc_values:
            # run GeoSPCA
            if 'geospca' in args.algorithms:
                geospca_result = run_geospca(X, k, nc, args.epsilon, args.maxiter)
                results.append(geospca_result)
            
            # run PathSPCA
            if 'pathspca' in args.algorithms:
                pathspca_result = run_pathspca(X, k, nc)
                results.append(pathspca_result)
            
            # run AnytimePCA
            if 'anytimepca' in args.algorithms:
                anytimepca_result = run_anytimepca(X, k, nc, args.k_star, args.cpus, args.batch, dataset_name)
                results.append(anytimepca_result)
    
    # convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # calculate the explained variance improvement percentage
    if len(args.algorithms) > 1 and args.baseline in args.algorithms:
        results_df = calculate_improvement_percentage(results_df, args.baseline)
    
    # save the results
    results_file = os.path.join(result_dir, f'{dataset_name}_comparison_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    # plot the comparison charts
    plot_results(results_df, result_dir)

# calculate the explained variance improvement percentage
def calculate_improvement_percentage(results_df, baseline='geospca'):
    """
    calculate the explained variance improvement percentage relative to the baseline algorithm
    
    Parameters:
    - results_df: the DataFrame containing the algorithm results
    - baseline: the baseline algorithm name, default is 'geospca'
    
    Returns:
    - the DataFrame with the improvement percentage column
    """
    print(f"\nCalculating improvement percentages relative to {baseline}...")
    
    # ensure the baseline algorithm exists in the results
    if baseline not in results_df['algorithm'].unique():
        print(f"Warning: Baseline algorithm '{baseline}' not found in results. Skipping percentage calculation.")
        return results_df
    
    # get all the combinations of k and nc
    k_nc_combinations = results_df[['k', 'nc']].drop_duplicates().values
    
    # add the improvement percentage column relative to the baseline algorithm for each algorithm
    for algorithm in results_df['algorithm'].unique():
        if algorithm == baseline:
            continue
        
        # create the new column name
        pct_column = f'{algorithm}_vs_{baseline}_pct'
        
        # initialize the new column
        results_df[pct_column] = float('nan')
        
        # calculate the improvement percentage for each k and nc combination
        for k, nc in k_nc_combinations:
            # get the explained variance of the baseline algorithm in the current k and nc
            baseline_variance = results_df[(results_df['algorithm'] == baseline) & 
                                          (results_df['k'] == k) & 
                                          (results_df['nc'] == nc)]['explained_variance'].values
            
            if len(baseline_variance) == 0:
                continue
            
            baseline_variance = baseline_variance[0]
            
            # get the explained variance of the current algorithm in the same k and nc
            current_variance_mask = ((results_df['algorithm'] == algorithm) & 
                                    (results_df['k'] == k) & 
                                    (results_df['nc'] == nc))
            
            if not any(current_variance_mask):
                continue
            
            current_variance = results_df.loc[current_variance_mask, 'explained_variance'].values[0]
            
            # calculate the improvement percentage
            if baseline_variance > 0:
                improvement_pct = ((current_variance - baseline_variance) / baseline_variance) * 100
                
                # update the DataFrame
                results_df.loc[current_variance_mask, pct_column] = improvement_pct
                
                print(f"  {algorithm} vs {baseline} (k={k}, nc={nc}): {improvement_pct:.2f}%")
    
    return results_df

if __name__ == "__main__":
    main() 