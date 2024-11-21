import numpy as np
import csv


# Define the activation function and its derivatives
def sigmoid(x):
    x = np.clip(x, -10, 10)  
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Data normalization
def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    std = np.where(std == 0, 1, std)  
    return (data - mean) / std, mean, std


# Loading Data Sets
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            bedrooms = int(row['bedrooms'])
            bathrooms = float(row['bathrooms'])
            sqft_lot = int(row['sqft_lot'])
            floors = float(row['floors'])
            sqft_above = int(row['sqft_above'])
            sqft_basement = int(row['sqft_basement'])
            view = int(row['view'])
            condition = int(row['condition'])
            price = float(row['price'])
            data.append([bedrooms, bathrooms, sqft_lot, floors, sqft_above, sqft_basement, view, condition, price])
    return np.array(data)


data = load_data('D://A1//data.csv')

