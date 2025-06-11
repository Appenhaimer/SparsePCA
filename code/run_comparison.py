import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Run Sparse PCA Algorithm Comparison')
    parser.add_argument('--dataset', type=str, choices=['creditcard', 'hars', 'covtype'], 
                        default='creditcard', help='Dataset to use')
    parser.add_argument('--small', action='store_true', help='Use smaller parameter settings for quick testing')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['geospca', 'pathspca', 'anytimepca'], 
                        choices=['geospca', 'pathspca', 'anytimepca'],
                        help='Algorithms to compare')
    args = parser.parse_args()
    
    # get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # get the project root directory
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # set the dataset path (use the absolute path)
    dataset_paths = {
        'creditcard': os.path.join(project_root, 'code', 'data', 'archive (2)', 'creditcard.csv'),
        'hars': os.path.join(project_root, 'code', 'data', 'archive (3)', 'train.csv'),
        'covtype': os.path.join(project_root, 'code', 'data', 'archive (4)', 'covtype.csv')
    }
    
    # build the comparison script path
    compare_script = os.path.join(current_dir, 'compare_algorithms.py')
    
    # build the command
    cmd = [sys.executable, compare_script, '--data', dataset_paths[args.dataset]]
    
    # if it is a small-scale test, use the smaller parameter settings
    if args.small:
        cmd.extend(['--k_values', '5', '10', 
                   '--nc_values', '2', 
                   '--maxiter', '100'])
    
    # add the algorithm parameters
    cmd.extend(['--algorithms'] + args.algorithms)
    
    print(f"Running command: {' '.join(cmd)}")
    
    # execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 