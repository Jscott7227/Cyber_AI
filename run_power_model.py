


import pandas as pd
import tensorflow as tf
import numpy as np
import keras_tuner as kt
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import subprocess
import argparse


def load_data(filepath):
    print('Loading Data -', filepath)
    return pd.read_csv(filepath)

def drop_columns(data):
    data['frame.time'] = data['frame.time'].str.replace(r'\sCDT$', '', regex=True)
    data['frame.time'] = pd.to_datetime(data['frame.time'])

    data = data.sort_values(by='frame.time')
    
    data = data.drop(columns=['frame.time', 'udp.srcport', 'udp.dstport', 'PLCID'])
    
    data = data.reset_index(drop=True)

    return data

def encode_data(data):
    unique_values = pd.Series(pd.concat([data['eth.src'], data['eth.dst']]).unique())
    encoding_map = {value: code for code, value in enumerate(unique_values)}
    data['eth.src'] = data['eth.src'].map(encoding_map)
    data['eth.dst'] = data['eth.dst'].map(encoding_map)
    unique_values = pd.Series(pd.concat([data['ip.src'], data['ip.dst']]).unique())
    encoding_map = {value: code for code, value in enumerate(unique_values)}
    data['ip.src'] = data['ip.src'].map(encoding_map)
    data['ip.dst'] = data['ip.dst'].map(encoding_map)
    data = pd.get_dummies(data, columns=['_ws.col.Protocol'])
    unique_values = pd.Series(pd.concat([data['tcp.srcport'], data['tcp.dstport']]).dropna().unique())
    encoding_map = {value: code for code, value in enumerate(unique_values)}
    data['tcp.srcport'] = data['tcp.srcport'].map(encoding_map).fillna(-1).astype(int)
    data['tcp.dstport'] = data['tcp.dstport'].map(encoding_map).fillna(-1).astype(int)
    data = data.fillna(-1)
    data = data.replace('-', 0)
    if '_ws.col.Protocol_UDP' in data.columns:
        data = data.drop(columns=['_ws.col.Protocol_UDP'])
    return data

def normalize_data(data):
    columns_to_norm = ['V1', 'I1', 'Theta', 'P', 'Q',
           '1_P', '1_Q', '2_P', '2_Q', '3_P', '3_Q',
           '4_P', '4_Q', '5_P', '5_Q', '6_P', '6_Q',
           '7_P', '7_Q', '9_P', '9_Q', '10_P', '10_Q',
           '11_P', '11_Q', '12_P', '12_Q', '13_P', '13_Q',
           '14_P', '14_Q']
    for col in columns_to_norm:
        data[col] = data[col].astype(str).str.replace(',', '').astype(np.float32)
        mean = data[col][data[col] != 0].mean()
        std = data[col][data[col] != 0].std()
        # Normalize and weight
        data[col] = data[col].apply(lambda x: (x - mean) / std if x != 0 else 0)
        data[col] = data[col] * 0.5

    return data

def calculate_S(P_col, Q_col):
    return np.sqrt(P_col**2 + Q_col**2)

def create_apparent_power(data):
    for i in range(1, 15):
        if i == 8:
            continue
        P_col = f'{i}_P'
        Q_col = f'{i}_Q'
        S_col = f'{i}_S'
        data[S_col] = calculate_S(data[P_col], data[Q_col])
    return data
    

def create_sequences(data, time_steps):
    X = []
    y = []
    data_X = np.delete(data, -1, axis=1)
    for i in range(len(data) - time_steps):
        row = [a for a in data_X[i:i+time_steps]]
        X.append(row)
        label = data[i+time_steps, -1]
        y.append(label)
    return np.array(X), np.array(y)


def clean_data(data):
    print("Preparing Data ...")
    data = drop_columns(data)
    data = encode_data(data)
    data = normalize_data(data)
    data = create_apparent_power(data)
    X, y = create_sequences(np.hstack([data.drop(columns=['label']), data['label'].values.reshape(-1, 1)]), time_steps=50)

    X = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('int32')

    return X, y

def load_model(model_path):
    print('Loading Model -', model_path)
    return tf.keras.models.load_model(model_path)

def model_run(X, y, model):
    print("Evaluating Model")
    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs > .5).astype(int)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, average=None, zero_division=0)
    
    print('Test Set Evaluation:')
    print(f'Accuracy: {accuracy:.4f}')
    for i in range(len(precision)):
        print(f"Class {i} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}")
    

if __name__ == "__main__": 

    print('Installing packages from requirements.txt')
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    parser = argparse.ArgumentParser(description="Inference script for cyber-physical anomaly detection competition")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the CSV file containing the test data")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model file (.pth or .h5)")
    args = parser.parse_args()
    data = load_data(args.data_path)
    X, y = clean_data(data)
    model = load_model(args.model_path)
    model_run(X, y, model)
    