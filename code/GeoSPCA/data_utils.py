import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path, features, zero_columns):
    df = pd.read_csv(data_path, header=0)
    for col in zero_columns:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].mean())
    A = df[features].to_numpy()
    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    return A, df 