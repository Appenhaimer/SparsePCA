import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from optimizer import optimize
from sklearn.preprocessing import StandardScaler
import contextlib

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 设置输出目录为项目根目录下的out文件夹
log_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'out')
os.makedirs(log_dir, exist_ok=True)

# 使用更详细的时间戳创建唯一的输出目录
timestamp = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
result_dir = os.path.join(log_dir, timestamp)
os.makedirs(result_dir, exist_ok=True)

log_file = os.path.join(result_dir, 'geospca_log.txt')
print(f"当前工作目录: {os.getcwd()}")
print(f"输出目录: {result_dir}")
print(f"日志文件: {log_file}")

# 确保输出目录存在
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"创建输出目录: {result_dir}")

sys.stdout = open(log_file, 'w', encoding='utf-8')
sys.stderr = sys.stdout

def load_dataset(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        df_numeric = df.select_dtypes(include=[np.number])
        X = df_numeric.values
    elif path.endswith('.data') or path.endswith('.dat') or path.endswith('.txt'):
        X = np.loadtxt(path)
    elif path.endswith('sonar.all-data'):
        df = pd.read_csv(path, header=None)
        X = df.iloc[:, :-1].values
    else:
        raise ValueError(f"Unknown file format: {path}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def run_geospca_for_ks(data_path, ks, nc, epsilon=0.1, maxiter=1000, out_dir=None):
    if out_dir is None:
        out_dir = result_dir
    
    print(f"\n开始处理数据集: {os.path.basename(data_path)}")
    print(f"结果将保存在: {out_dir}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")
    
    X = load_dataset(data_path)
    print(f"数据集加载完成，维度: {X.shape}")
    
    dA = np.array([np.linalg.norm(X[:, i]) ** 2 for i in range(X.shape[1])])
    total_variance = np.sum(dA)  # 计算总方差
    for k in ks:
        print(f"\n正在处理 k={k}, nc={nc} 的情况...")
        log_file = os.path.join(out_dir, f'geospca_{os.path.splitext(os.path.basename(data_path))[0]}_k{k}_nc{nc}.log')
        with open(log_file, 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            print(f"Processing {data_path} with k={k}, nc={nc}, epsilon={epsilon}, maxiter={maxiter}")
            try:
                Bindices, Bvalue, iteration = optimize(X, dA, nc, k, epsilon, maxiter)
                explained_ratio = Bvalue / total_variance if total_variance > 0 else 0
                print('Result:')
                result = {
                    'dataset': os.path.basename(data_path),
                    'selected_indices': Bindices,
                    'explained_variance': Bvalue,
                    'explained_ratio': explained_ratio,
                    'iterations': iteration,
                    'k': k,
                    'nc': nc,
                    'epsilon': epsilon,
                    'maxiter': maxiter
                }
                print(result)
                
                out_file = os.path.join(out_dir, f'geospca_{os.path.splitext(os.path.basename(data_path))[0]}_k{k}_nc{nc}.csv')
                pd.DataFrame([result]).to_csv(out_file, index=False)
                print(f"\nk={k}, nc={nc} 的结果已保存到: {out_file}")
            except Exception as e:
                print(f"处理 k={k}, nc={nc} 时发生错误: {e}")
                raise e
    
    print(f"\n所有处理完成！结果文件保存在: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--k', type=int, nargs='+', required=True, help='List of k values, e.g. --k 3 5 7')
    parser.add_argument('--nc', type=int, nargs='+', required=True, help='List of nc values, e.g. --nc 2 4 6')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value')
    parser.add_argument('--maxiter', type=int, default=1000, help='Max iterations')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    try:
        # 对每个nc值分别运行
        for nc in args.nc:
            run_geospca_for_ks(
                data_path=args.data,
                ks=args.k,
                nc=nc,
                epsilon=args.epsilon,
                maxiter=args.maxiter,
                out_dir=args.out_dir
            )
    except Exception as e:
        print(f"程序运行出错: {e}")
        raise e 