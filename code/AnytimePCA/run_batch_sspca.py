import os
import subprocess
import time
from datetime import datetime
import glob
import pandas as pd

def run_sspca_experiments(dataset_path, k_values, k_star_values, cpus=4, batch=210):
    """
    运行多个参数组合的SSPCA实验
    
    参数:
    dataset_path: 数据集路径
    k_values: 要测试的k值列表
    k_star_values: 要测试的k_star值列表
    cpus: CPU数量
    batch: 批处理大小
    """
    # 获取数据集名称
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    result_dir = os.path.join('out', f'batch_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(result_dir, 'batch_run_log.txt')
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行所有参数组合
    total_experiments = len(k_values) * len(k_star_values)
    completed = 0
    
    with open(log_file, 'w') as f:
        f.write(f"Starting batch experiments for {dataset_name}\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for k in k_values:
            for k_star in k_star_values:
                # 构建命令
                cmd = [
                    'python', 'code/AnytimePCA/sspca_exp.py',
                    '--k', str(k),
                    '--k_star', str(k_star),
                    '--fpath', dataset_path,
                    '--cpus', str(cpus),
                    '--batch', str(batch),
                    '--newrun', '1'
                ]
                
                # 记录实验开始
                experiment_start = time.time()
                f.write(f"\nRunning experiment: k={k}, k_star={k_star}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                
                try:
                    # 运行命令
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # 记录结果
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write(f"Output: {result.stdout}\n")
                    if result.stderr:
                        f.write(f"Error: {result.stderr}\n")
                    
                    # 查找最新生成的结果文件
                    date_dir = datetime.now().strftime("%y_%m_%d")
                    result_files = glob.glob(f"out/{date_dir}/sspcaExp_*_kstar{k_star}_k{k}_*.csv")
                    if result_files:
                        latest_file = max(result_files, key=os.path.getctime)
                        try:
                            df = pd.read_csv(latest_file)
                            ev = df.get('explained_variance', [None])[0]
                            ratio = df.get('explained_ratio', [None])[0]
                            f.write(f"Explained Variance: {ev}\n")
                            f.write(f"Explained Variance Ratio: {ratio}\n")
                        except Exception as e:
                            f.write(f"Error reading result file: {e}\n")
                    else:
                        f.write("Result file not found.\n")
                    
                    completed += 1
                    experiment_time = time.time() - experiment_start
                    f.write(f"Experiment completed in {experiment_time:.2f} seconds\n")
                    f.write(f"Progress: {completed}/{total_experiments} experiments completed\n")
                    
                except Exception as e:
                    f.write(f"Error running experiment: {str(e)}\n")
                
                f.write("-" * 80 + "\n")
                f.flush()
        
        # 记录总运行时间
        total_time = time.time() - start_time
        f.write(f"\nBatch run completed in {total_time:.2f} seconds\n")
        f.write(f"Completed {completed}/{total_experiments} experiments\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    # 运行train.csv的实验
    dataset_path = "data/archive (2)/creditcard.csv"
    k_values = [5, 10, 15, 20]  # 与geospca实验相同的k值
    k_star_values = [1]  # 对应geospca的nc参数
    
    run_sspca_experiments(
        dataset_path=dataset_path,
        k_values=k_values,
        k_star_values=k_star_values,
        cpus=4,
        batch=210
    ) 