from data_utils import load_and_preprocess_data
from optimizer import optimize
import numpy as np

# 参数设置
def main():
    nc = 4
    k = 6
    epsilon = 0.1
    maxiter = 1000
    data = 'archive/diabetes.csv'
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # 数据加载
    A, df = load_and_preprocess_data(data, features, zero_columns)
    n, p = A.shape
    dA = np.array([np.linalg.norm(A[:, i]) ** 2 for i in range(p)])

    # 优化
    Bindices, Bvalue, iteration = optimize(A, dA, nc, k, epsilon, maxiter)

    print("Best solution found")
    print(f'Explained Variance = {Bvalue:.4f}')
    print("Selected feature indices:", Bindices)
    print("Selected features:", [features[i] for i in Bindices])
    print(f'Number of iterations = {iteration}')

if __name__ == "__main__":
    main() 