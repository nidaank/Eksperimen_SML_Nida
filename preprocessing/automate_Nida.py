from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import dump
import pandas as pd
import numpy as np

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def handle_outliers(data, outliers):
    median = data.median()
    data[outliers] = median
    return data

def preprocess_data(data, target_col, scaler_save_path, output_csv_path):
    # 1. Hapus data duplikat dan missing values
    data = data.drop_duplicates()
    data = data.dropna()
    data = data.reset_index(drop=True)

    # 2. Pisahkan fitur numerik dan kategorikal (kecuali target)
    numeric_features = data.drop(columns=[target_col]).select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.drop(columns=[target_col]).select_dtypes(include=['object']).columns.tolist()

    # 3. Tangani outlier di fitur numerik
    for col in numeric_features:
        outliers = detect_outliers_iqr(data[col])
        data[col] = handle_outliers(data[col], outliers)

    # 4. Normalisasi numerik
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    dump(scaler, scaler_save_path)
    print(f"Scaler berhasil disimpan ke: {scaler_save_path}")

    # 5. Label Encoding fitur kategorikal
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le

    # 6. Label encoding target jika kategorikal
    if data[target_col].dtype == 'object':
        le = LabelEncoder()
        data[target_col] = le.fit_transform(data[target_col])
        le_dict['__target__'] = le

    # 7. Simpan hasil akhir ke CSV
    data.to_csv(output_csv_path, index=False)
    print(f"Data hasil preprocessing berhasil disimpan ke: {output_csv_path}")