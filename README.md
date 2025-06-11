# SparsePCA
Performance Comparison of GeoSPCA, AnytimePCA and PathSPCA
with experimental results analysis (plots) generated

##License and Attribution
This project includes AnytimePCA implementation from original repository https://github.com/sdannyvi/AnytimePCA/tree/master. The original implementation is licensed under MIT License. Please refer to the original repository for more details.

This project includes GeoSPCA implementation from original repository https://github.com/DLKitane/GEOSPCA. The original implementation is licensed under MIT License. Please refer to the original repository for more details.

## Dataset(not uploaded): 
1. Credit Card Fraud Detection (creditcard)
2. Human Activity Recognition with Smartphones (hars)
3. Forest Cover Type (covtype)

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_comparison.py [options]
```

### Options
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

