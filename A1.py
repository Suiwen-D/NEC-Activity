import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


# Define the activation function and its derivatives
def sigmoid(x):
    x = np.clip(x, -10, 10)  # prevent spillage
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Data normalization
def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    std = np.where(std == 0, 1, std)  # Prevent dividing by 0
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
            yr_built = int(row['yr_built'])
            yr_renovated = int(row['yr_renovated']) if row['yr_renovated'] else 0
            sqft_above = int(row['sqft_above'])
            sqft_basement = int(row['sqft_basement'])
            view = int(row['view'])
            condition = int(row['condition'])
            price = float(row['price'])
            data.append([bedrooms, bathrooms, sqft_lot, floors, yr_built, yr_renovated, sqft_above, sqft_basement, view, condition, price])
    return np.array(data)


# Partitioning data sets
def split_data(data, train_ratio=0.8, val_ratio=0.1):
    np.random.shuffle(data)
    total_samples = data.shape[0]
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    return train_data, val_data, test_data


# the Neural Network Class
class NeuralNet:
    def __init__(self, layers, learning_rate=0.0001, momentum=0.9, l2_reg=0.001):
        self.layers = layers
        self.L = len(layers)  # Add L to represent the number of layers
        self.n = layers.copy()  
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.weights = []
        self.biases = []
        self.xi = []  
        self.h = []  
        self.deltas = [] 
        self.d_w = []  
        self.d_theta = [] 
        self.prev_weight_changes = []
        self.prev_bias_changes = []
        self.initialize_parameters()

    def initialize_parameters(self):
        for i in range(1, self.L):
            weight = np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(2 / self.layers[i - 1])  # He初始化
            bias = np.zeros((self.layers[i], 1))
            self.weights.append(weight)
            self.biases.append(bias)
            self.d_w.append(np.zeros_like(weight))
            self.d_theta.append(np.zeros_like(bias))
            self.prev_weight_changes.append(np.zeros_like(weight))
            self.prev_bias_changes.append(np.zeros_like(bias))

    def forward_propagation(self, X):
        self.xi = [X.T]  # Initialization xi list
        self.h = []
        activations = [X.T]
        weighted_sums = []
        for i in range(len(self.weights)):
            weighted_sum = np.dot(self.weights[i], activations[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            self.h.append(weighted_sum)  # Store the value of h
            if i == len(self.weights) - 1:  # The output layer uses linear activation
                activation = weighted_sum
            else:
                activation = sigmoid(weighted_sum)
            activations.append(activation)
            self.xi.append(activation)  # Store the value of xi
        return activations, weighted_sums

    def backward_propagation(self, activations, weighted_sums, y):
        self.deltas = []  # Reinitialize the deltas list
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:  # output layer
                error = activations[i + 1] - y
                self.deltas.append(error)
            else:
                delta = np.dot(self.weights[i + 1].T, self.deltas[-1]) * sigmoid_derivative(activations[i + 1])
                self.deltas.append(delta)
        self.deltas.reverse()
        for i in range(len(self.weights)):
            self.d_w[i] = (
                -self.learning_rate * np.dot(self.deltas[i], self.xi[i].T)
                + self.momentum * self.prev_weight_changes[i]
                - self.l2_reg * self.weights[i]  # L2 Regularization
            )
            self.d_theta[i] = -self.learning_rate * np.sum(self.deltas[i], axis=1, keepdims=True) + self.momentum * self.prev_bias_changes[i]
            self.weights[i] += self.d_w[i]
            self.biases[i] += self.d_theta[i]
            self.prev_weight_changes[i] = self.d_w[i]
            self.prev_bias_changes[i] = self.d_theta[i]

    def train(self, X, y, epochs, val_data=None):
        train_errors = []
        val_errors = []
        for epoch in range(epochs):
            activations, weighted_sums = self.forward_propagation(X)
            self.backward_propagation(activations, weighted_sums, y.T)
            train_error = self.calculate_error(activations[-1], y.T)
            train_errors.append(train_error)
            if val_data is not None:
                val_error = self.validate(val_data[0], val_data[1])
                val_errors.append(val_error)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Error: {train_error}, Val Error: {val_error if val_data is not None else ""}')
        return train_errors, val_errors

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1].T

    def calculate_error(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def validate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        return self.calculate_error(y_pred, y_val)


# Loading Data Sets
data = load_data('D://PyProject//A1//data.csv')
train_data, val_data, test_data = split_data(data)

# Separating Features and Target Variables
X_train, y_train = train_data[:, :-1], train_data[:, -1].reshape(-1, 1)
X_val, y_val = val_data[:, :-1], val_data[:, -1].reshape(-1, 1)
X_test, y_test = test_data[:, :-1], test_data[:, -1].reshape(-1, 1)

# Data normalization
X_train_normalized, X_mean, X_std = normalize_data(X_train)
X_val_normalized, _, _ = normalize_data(X_val, X_mean, X_std)
X_test_normalized, _, _ = normalize_data(X_test, X_mean, X_std)

# Target variable normalization
y_train_normalized, y_mean, y_std = normalize_data(y_train)
y_val_normalized, _, _ = normalize_data(y_val, y_mean, y_std)
y_test_normalized, _, _ = normalize_data(y_test, y_mean, y_std)

# Creating Neural Network Models
model = NeuralNet(layers=[X_train_normalized.shape[1], 10, 1], learning_rate=0.0001, momentum=0.9, l2_reg=0.001)

# train the model
train_errors, val_errors = model.train(X_train_normalized, y_train_normalized, epochs=1000, val_data=(X_val_normalized, y_val_normalized))





# Predicting and restoring data
y_pred_normalized = model.predict(X_test_normalized)
y_pred = y_pred_normalized * y_std + y_mean

# Calculating Test Errors
test_error = model.calculate_error(y_pred, y_test)
print(f"Test Error: {test_error}")



# Restore Real and Predicted Values
y_test_original = y_test * y_std + y_mean
y_pred_original = y_pred

#Plotting Scatter Points
# import matplotlib.pyplot as plt

# plt.scatter(y_test_original, y_pred_original)
# plt.xlabel('True Values (Original Scale)')
# plt.ylabel('Predicted Values')
# plt.title('True vs Predicted Values')
# plt.show()



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Logarithmic transformation of target values
y_train_log = np.log1p(y_train)  # 1 + log(x)
y_test_log = np.log1p(y_test)
y_val_log = np.log1p(y_val)

# Training Model after Logarithmic Transformation
model = NeuralNet(layers=[X_train_normalized.shape[1], 10, 1], learning_rate=0.0001, momentum=0.9, l2_reg=0.001)
train_errors, val_errors = model.train(X_train_normalized, y_train_log, epochs=1000, val_data=(X_val_normalized, y_val_log))

# Predicting and Inverting Transformations
y_pred_log = model.predict(X_test_normalized)
y_pred = np.expm1(y_pred_log)  # Restore to Original Space

# Calculating Test Errors
test_error = mean_squared_error(y_test, y_pred)
print(f"Test Error (MSE): {test_error}")

# Fix computation problems with MAP
def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true > 1e-8  # Ignore values near zero
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mse, mae, mape

# Output target value range
print(f"Target Value Range: Min={y_train.min()}, Max={y_train.max()}")




#Hyperparametric Comparison and Selection
hyperparameters = [
    {'layers': [X_train_normalized.shape[1], 10, 1], 'epochs': 1000, 'lr': 0.0001, 'momentum': 0.9},
    {'layers': [X_train_normalized.shape[1], 20, 10, 1], 'epochs': 1000, 'lr': 0.0001, 'momentum': 0.8},
    # Add more superparametric combinations...
]

results = []
best_train_errors, best_val_errors = None, None  # Training/Validation Errors for Storing Optimal Models
for params in hyperparameters:
    model = NeuralNet(
        layers=params['layers'],
        learning_rate=params['lr'],
        momentum=params['momentum'],
        l2_reg=0.001
    )
    train_errors, val_errors = model.train(
        X_train_normalized, y_train_normalized, epochs=params['epochs'], val_data=(X_val_normalized, y_val_normalized)
    )
    y_pred = model.predict(X_test_normalized) * y_std + y_mean
    mse, mae, mape = evaluate_metrics(y_test, y_pred)
    results.append({'params': params, 'mse': mse, 'mae': mae, 'mape': mape})
    if not best_train_errors or mse < min(result['mse'] for result in results):
        best_train_errors, best_val_errors = train_errors, val_errors

# output
print("Hyperparameter Results:")
for result in results:
    print(f"Params: {result['params']}, MSE: {result['mse']:.4f}, MAE: {result['mae']:.4f}, MAPE: {result['mape']:.2f}%")

# Drawing Training and Validation Errors with epoch
if best_train_errors and best_val_errors:  # Ensure Error Data Exists
    plt.plot(best_train_errors, label='Train Error')
    plt.plot(best_val_errors, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Error')
    plt.legend()
    plt.show()

# Model Results Comparison
# 1. multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train_normalized, y_train_normalized)
y_pred_mlr_normalized = mlr.predict(X_test_normalized)
y_pred_mlr = y_pred_mlr_normalized * y_std + y_mean
mse_mlr, mae_mlr, mape_mlr = evaluate_metrics(y_test, y_pred_mlr)

# 2. Neural Networks Using Open Source Libraries
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, learning_rate_init=0.001, random_state=42)
mlp.fit(X_train_normalized, y_train_normalized.ravel())
y_pred_mlp_normalized = mlp.predict(X_test_normalized)
y_pred_mlp = y_pred_mlp_normalized * y_std + y_mean
mse_mlp, mae_mlp, mape_mlp = evaluate_metrics(y_test, y_pred_mlp)

# Compare three models
print("\nModel Comparison:")
print(f"Your Neural Network - MSE: {test_error:.4f}, MAE: {mae_mlr:.4f}, MAPE: {mape_mlr:.2f}%")
print(f"Linear Regression - MSE: {mse_mlr:.4f}, MAE: {mae_mlr:.4f}, MAPE: {mape_mlr:.2f}%")
print(f"MLP Regressor - MSE: {mse_mlp:.4f}, MAE: {mae_mlp:.4f}, MAPE: {mape_mlp:.2f}%")

# Plotting Scatter Points
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Your Neural Network")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_mlr, alpha=0.6)
plt.title("Linear Regression")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_mlp, alpha=0.6)
plt.title("MLP Regressor")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()
