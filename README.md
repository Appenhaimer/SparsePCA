# Language

- [English](#english)
- [中文](#中文)


# English
# SparsePCA
Performance Comparison of GeoSPCA, AnytimePCA and PathSPCA
with experimental results analysis (plots) generated

##License and Attribution
This project includes AnytimePCA implementation from original repository https://github.com/sdannyvi/AnytimePCA/tree/master. The original implementation is licensed under MIT License. Please refer to the original repository for more details.

This project includes GeoSPCA implementation from original repository https://github.com/DLKitane/GEOSPCA. The original implementation is licensed under MIT License. Please refer to the original repository for more details.

## Dataset: 
1. Credit Card Fraud Detection (creditcard)
   - Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Place the `creditcard.csv` file in `code/data/archive (2)/` directory

2. Human Activity Recognition with Smartphones (hars)
   - Download from: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
   - Place the `train.csv` file in `code/data/archive (3)/` directory

3. Forest Cover Type (covtype)
   - Download from: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
   - Place the `covtype.csv` file in `code/data/archive (4)/` directory

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_comparison.py [options]
```

## Usage Options
- `--dataset`: Choose dataset to use (default: creditcard)
  - Available datasets: creditcard, hars, covtype
- `--small`: Use smaller parameter settings for quick testing
- `--algorithms`: Specify algorithms to compare (default: all)
  - Available algorithms: geospca, pathspca, anytimepca

### Examples
```bash
# Run with default settings (creditcard dataset, all algorithms)
python run_comparison.py

# Run with small parameter settings for quick testing
python run_comparison.py --small

# Run specific algorithms on a specific dataset
python run_comparison.py --dataset hars --algorithms geospca pathspca
```

## Paper Reference:

@inproceedings{holtzman2020greedy,
  title={A greedy anytime algorithm for sparse PCA},
  author={Holtzman, Guy and Soffer, Adam and Vilenchik, Dan},
  booktitle={Conference on Learning Theory},
  pages={1939--1956},
  year={2020},
  organization={PMLR}
}

@article{d2008optimal,
  title={Optimal solutions for sparse principal component analysis.},
  author={d'Aspremont, Alexandre and Bach, Francis and El Ghaoui, Laurent},
  journal={Journal of Machine Learning Research},
  volume={9},
  number={7},
  year={2008}
}

# 中文

# SparsePCA
GeoSPCA、AnytimePCA 和 PathSPCA 的性能比较
包含实验结果分析和可视化图表

## 版权和引用说明
本项目包含来自原始仓库 https://github.com/sdannyvi/AnytimePCA/tree/master 的 AnytimePCA 实现。原始实现采用 MIT 许可证。更多详情请参考原始仓库。

本项目包含来自原始仓库 https://github.com/DLKitane/GEOSPCA 的 GeoSPCA 实现。原始实现采用 MIT 许可证。更多详情请参考原始仓库。

## 数据集：
1. Credit Card Fraud Detection (creditcard) (creditcard)
   - 下载地址：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - 将 `creditcard.csv` 文件放置在 `code/data/archive (2)/` 目录下

2. Human Activity Recognition with Smartphones (hars)
   - Download from: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
   - Place the `train.csv` file in `code/data/archive (3)/` directory

3. Forest Cover Type (covtype)
   - 下载地址：https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
   - 将 `covtype.csv` 文件放置在 `code/data/archive (4)/` 目录下

## 安装
```bash
pip install -r requirements.txt
```

## 运行
```bash
python run_comparison.py [option]
```

## 使用选项
- `--dataset`: 选择要使用的数据集（默认：creditcard）
  - 可选数据集：creditcard, hars, covtype
- `--small`: 使用较小的参数设置进行快速测试
- `--algorithms`: 指定要比较的算法（默认：全部）
  - 可选算法：geospca, pathspca, anytimepca

### 使用示例
```bash
# 使用默认设置运行（creditcard数据集，所有算法）
python run_comparison.py

# 使用小规模参数设置进行快速测试
python run_comparison.py --small

# 在特定数据集上运行特定算法
python run_comparison.py --dataset hars --algorithms geospca pathspca
```

## 论文引用：

@inproceedings{holtzman2020greedy,
  title={A greedy anytime algorithm for sparse PCA},
  author={Holtzman, Guy and Soffer, Adam and Vilenchik, Dan},
  booktitle={Conference on Learning Theory},
  pages={1939--1956},
  year={2020},
  organization={PMLR}
}

@article{d2008optimal,
  title={Optimal solutions for sparse principal component analysis.},
  author={d'Aspremont, Alexandre and Bach, Francis and El Ghaoui, Laurent},
  journal={Journal of Machine Learning Research},
  volume={9},
  number={7},
  year={2008}
}

