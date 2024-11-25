


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
    return pd.read_csv(filepath)

def drop_columns(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    data = data.sort_values(by='timestamp')
    
    data = data.drop(columns=['timestamp', 'timestamp_c', 'frame.number', 'frame.len', 'wlan_radio.Noise level (dbm)', 'wlan_radio.SNR (db)', 'wlan.frag', 'wlan.qos.ack', 'mpitch.1', 'mroll.1', 'myaw.1'])
    
    data = data.reset_index(drop=True)

    return data

def weight_data(data):
    physical_columns = [
    'mid', 'x', 'y', 'z', 'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 
    'templ', 'temph', 'tof', 'h', 'bat', 'baro', 'flight_time', 'agx', 'agy', 'agz'
    ]

    weight_factor = .5
    data[physical_columns] = data[physical_columns] * weight_factor
    
    return data

def normalize_data(df):
    continuous_cols = ['x', 'y', 'z', 'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'bat', 'flight_time', 'baro', 'templ', 'temph', 'tof', 'h',
                      'agx', 'agy', 'agz', 'wlan.fcs']
    scaler = MinMaxScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    return df 

def create_sequences(data, time_steps):
    X = []
    y = []
    for i in range(len(data) - time_steps):
        row = [a for a in data[i:i+time_steps]]
        X.append(row)
        label = data[i+time_steps, -1]
        y.append(label)
    return np.array(X), np.array(y)


def clean_data(df):
    df = drop_columns(df)
    df = weight_data(df)
    df = normalize_data(df)
    X, y = create_sequences(np.hstack([df, df['class'].values.reshape(-1, 1)]), time_steps=5)

    return X, y

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
def model_run(X, y, model):
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

    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    parser = argparse.ArgumentParser(description="Inference script for cyber-physical anomaly detection competition")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the CSV file containing the test data")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model file (.pth or .h5)")
    args = parser.parse_args()
    data = load_data(args.data_path)
    X, y = clean_data(data)
    model = load_model(args.model_path)
    model_run(X, y, model)
    