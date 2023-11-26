import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to load data
def load_data(city):
    file_path = filedialog.askopenfilename(title=f"Select {city} CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        return pd.read_csv(file_path)
    return None

# Function to preprocess data
def preprocess_data(dataset):
    dataset.dropna(inplace=True)
    dataset['YEAR'] = pd.to_datetime(dataset['YEAR']).dt.year
    dataset['MONTH'] = pd.to_datetime(dataset['MONTH']).dt.month
    dataset['DAY'] = pd.to_datetime(dataset['DAY']).dt.day
    return dataset

# Function to train Random Forest model
def train_random_forest_model(x_train, y_train):
    param_grid = {'n_estimators': [100, 500, 1000],
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    regressor = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                      max_depth=best_params['max_depth'],
                                      min_samples_split=best_params['min_samples_split'],
                                      random_state=0)
    regressor.fit(x_train, y_train)
    return regressor

# Function to train Neural Network model using PyTorch
def train_neural_network_model(x_train, y_train):
    input_size = x_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)

    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(x_train_tensor).squeeze()
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model

# Function to predict temperature using Neural Network
def predict_temperature_neural_network(model, scaler, input_date):
    input_year, input_month, input_day = map(int, input_date.split('-'))
    input_features = np.array([[input_year, input_month, input_day]])
    input_features = scaler.transform(input_features)
    input_tensor = torch.FloatTensor(input_features)
    predicted_temperature = model(input_tensor).item()
    return (predicted_temperature - 32) * 5 / 9

# GUI setup
root = tk.Tk()
root.title("Temperature Prediction")

datasets = {}
for city in cities:
    dataset = load_data(city)
    datasets[city] = dataset

# User Input for Date
input_date = input("Enter the date (YYYY-MM-DD): ")

# Predict Temperature for Input Date in each City using Random Forest
for city in cities:
    dataset = datasets[city]
    dataset = preprocess_data(dataset)

    features = dataset[['YEAR', 'MONTH', 'DAY']].values
    target = dataset['TEMPERATURE'].values

    scaler_rf = StandardScaler()
    features_rf = scaler_rf.fit_transform(features)

    x_train_rf, _, y_train_rf, _ = train_test_split(features_rf, target, test_size=0.3, random_state=0)

    regressor_rf = train_random_forest_model(x_train_rf, y_train_rf)

    predicted_temperature_rf = predict_temperature(regressor_rf, scaler_rf, input_date)


# Predict Temperature for Input Date in each City using Neural Network
for city in cities:
    dataset = datasets[city]
    dataset = preprocess_data(dataset)

    features_nn = dataset[['YEAR', 'MONTH', 'DAY']].values
    target_nn = dataset['TEMPERATURE'].values

    scaler_nn = MinMaxScaler()
    features_nn = scaler_nn.fit_transform(features_nn)

    x_train_nn, _, y_train_nn, _ = train_test_split(features_nn, target_nn, test_size=0.3, random_state=0)

    model_nn = train_neural_network_model(x_train_nn, y_train_nn)

    predicted_temperature_nn = predict_temperature_neural_network(model_nn, scaler_nn, input_date)

    print(f"Predicted Mean Temperature for {city} on {input_date} (Neural Network): {predicted_temperature_nn:.2f}°C")

sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
for city in cities:
    sns.lineplot(x='YEAR', y='TEMPERATURE', data=datasets[city], label=city)

plt.title('Temperature Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

root.mainloop()
