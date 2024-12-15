import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



# Define the activation function and its derivatives
def sigmoid(x):
    x = np.clip(x, -10, 10)  # prevent spillage （OverflowError）
    return 1 / (1 + np.exp(-x))

#Calculate the derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Data normalization（Speed up convergence of gradient descent algorithm）
def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0) #Calculates the mean of each feature by column.
        std = np.std(data, axis=0) #Calculates the standard deviation for each feature by column.
    std = np.where(std == 0, 1, std)  # Prevent dividing by 0
    return (data - mean) / std, mean, std # Normalized data


# Loading Data Sets， convert it to a numerical matrix.
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
            yr_renovated = int(row['yr_renovated']) if row['yr_renovated'] else 0 #Some fields may be null and require special processing
            sqft_above = int(row['sqft_above'])
            sqft_basement = int(row['sqft_basement'])
            view = int(row['view'])
            condition = int(row['condition'])
            price = float(row['price'])
            data.append([bedrooms, bathrooms, sqft_lot, floors, yr_built, yr_renovated, sqft_above, sqft_basement, view, condition, price])
    return np.array(data) #Returns as a NumPy array for easy subsequent processing.


# Partitioning data sets
def split_data(data, train_ratio=0.8, val_ratio=0.1): #80% for training, 10% for validation, and 10% for testing.
    np.random.shuffle(data) #Randomly disrupt data to avoid sequential deviations in results
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
        self.layers = layers #Stores the number of nodes per layer, in the form of a list.
        self.L = len(layers)  # Represents the total number of layers L of the network, which is used when forward propagation, reverse propagation, etc. needs to traverse each layer
        self.n = layers.copy()  #Copy the layers list to self.n. The purpose is to explicitly ensure the number of nodes per layer.
        self.learning_rate = learning_rate
        self.momentum = momentum #Accelerate convergence and reduce shocks
        self.l2_reg = l2_reg #L2 regularizes the coefficient to control the size of the weight and prevent the model from overfitting.
        self.weights = []
        self.biases = []
        #These two lists are cached in dynamic network calculations to avoid double calculations
        self.xi = []  
        self.h = []  
        #Stores error terms for each layer in reverse propagation
        self.deltas = [] 
        #These lists cache gradient information for updates of weights and biases.
        self.d_w = []  
        self.d_theta = [] 

        self.prev_weight_changes = []#Stores the value of the last round of weight updates
        self.prev_bias_changes = []#Stores values of the last round of bias updates
        self.initialize_parameters()#Initialize the parameters of the model, including weights and biases.

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
        self.h = [] #Initialize an empty list self.h, which will be used to store each layer of neuron computation resulting in a weighted sum
        activations = [X.T] #This list is used to store the activation values (output values) of each layer of the neural network processed by the activation function
        weighted_sums = [] #Create an empty list to store the weights and results calculated by each layer of neuron
        for i in range(len(self.weights)): #Layer by layer calculate the weighted sum
            # get the weighted sum and then add to the list.
            weighted_sum = np.dot(self.weights[i], activations[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            self.h.append(weighted_sum)  # Store the value of h
            if i == len(self.weights) - 1:  # Determine whether the current layer is an output layer (by determining whether it is a layer corresponding to the last layer weight, since the length of self.weights represents the total number of layers minus 1)
                activation = weighted_sum #Calculate the activation value
            else:
                activation = sigmoid(weighted_sum)#If not the output layer, apply the sigmoid activation function
            activations.append(activation)
            self.xi.append(activation)  # Store the value of xi for backward propagation
        return activations, weighted_sums #Weighted Z containing all layers to calculate activation values or backward propagation

    def backward_propagation(self, activations, weighted_sums, y):
        self.deltas = []  # Reinitialize the deltas list
        for i in reversed(range(len(self.weights))): #self.deltas for storing error terms for each layer in backward propagation
            if i == len(self.weights) - 1:  # output layer
                #Get the output layer error and add it to the self.deltas list.
                error = activations[i + 1] - y
                self.deltas.append(error)
            else:
                #Based on information such as error terms at the next layer and activation values at the current layer
                delta = np.dot(self.weights[i + 1].T, self.deltas[-1]) * sigmoid_derivative(activations[i + 1])
                self.deltas.append(delta)
        self.deltas.reverse() #Reversing Error Item List Order
        for i in range(len(self.weights)): #Update weights and biases
            
            #Optional Part 1: Study the effect of the different regularization techniques in the Neural Network

            self.d_w[i] = (
                #Calculate the updating direction and step length of the weight according to gradient descent
                -self.learning_rate * np.dot(self.deltas[i], self.xi[i].T)
                #Accelerate convergence by multiplying the amount of change from the previous round of weight updates by the momentum coefficient
                + self.momentum * self.prev_weight_changes[i]
                - self.l2_reg * self.weights[i]  # L2 Regularization,This item will cause the weight to be updated in the direction of zero
            )
            #Calculate the amount of update for the bias
            self.d_theta[i] = -self.learning_rate * np.sum(self.deltas[i], axis=1, keepdims=True) + self.momentum * self.prev_bias_changes[i]
            #Update weights and biases
            self.weights[i] += self.d_w[i]
            self.biases[i] += self.d_theta[i]
            #Recording changes in weights and biases
            self.prev_weight_changes[i] = self.d_w[i]
            self.prev_bias_changes[i] = self.d_theta[i]

    def train(self, X, y, epochs, val_data=None):
        train_errors = [] #Stores the error of each training set round.
        val_errors = [] #Store errors for each round of validation sets
        for epoch in range(epochs):
            activations, weighted_sums = self.forward_propagation(X)
            self.backward_propagation(activations, weighted_sums, y.T)
            #Calculate the error between the predicted and true values by self.calculate_error and save
            train_error = self.calculate_error(activations[-1], y.T)
            train_errors.append(train_error)
            if val_data is not None:
                val_error = self.validate(val_data[0], val_data[1])
                val_errors.append(val_error)
            if epoch % 100 == 0:
                #Training errors and validation errors are printed every 100 rounds
                print(f'Epoch {epoch}, Train Error: {train_error}, Val Error: {val_error if val_data is not None else ""}')
        return train_errors, val_errors

#Predicting output of input data
    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1].T
#Calculate the error between the prediction and the true result
    def calculate_error(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
#Evaluate the performance of the model on the validation set by calculating the error of the validation set.
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
# test_error = model.calculate_error(y_pred, y_test)
# print(f"Test Error: {test_error}")



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

# # Logarithmic transformation of target values
# y_train_log = np.log1p(y_train)  # 1 + log(x)
# y_test_log = np.log1p(y_test)
# y_val_log = np.log1p(y_val)

# # Training Model after Logarithmic Transformation
# model = NeuralNet(layers=[X_train_normalized.shape[1], 10, 1], learning_rate=0.0001, momentum=0.9, l2_reg=0.001)
# train_errors, val_errors = model.train(X_train_normalized, y_train_log, epochs=1000, val_data=(X_val_normalized, y_val_log))

# # Predicting and Inverting Transformations
# y_pred_log = model.predict(X_test_normalized)
# y_pred = np.expm1(y_pred_log)  # Restore to Original Space

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




#Part 3.1 Hyperparametric Comparison and Selection

hyperparameters = [
    {'layers': [X_train_normalized.shape[1], 10, 1], 'epochs': 500, 'lr': 0.0001, 'momentum': 0.9},
    {'layers': [X_train_normalized.shape[1], 20, 10, 1], 'epochs': 600, 'lr': 0.0001, 'momentum': 0.8},
    {'layers': [X_train_normalized.shape[1], 50, 25, 10, 1], 'epochs': 700, 'lr': 0.00005, 'momentum': 0.9},
    {'layers': [X_train_normalized.shape[1], 30, 15, 1], 'epochs': 800, 'lr': 0.0001, 'momentum': 0.7},
    {'layers': [X_train_normalized.shape[1], 15, 5, 1], 'epochs': 900, 'lr': 0.0002, 'momentum': 0.85},
    {'layers': [X_train_normalized.shape[1], 40, 20, 10, 1], 'epochs': 500, 'lr': 0.00005, 'momentum': 0.8},
    {'layers': [X_train_normalized.shape[1], 25, 10, 1], 'epochs': 600, 'lr': 0.0001, 'momentum': 0.95},
    {'layers': [X_train_normalized.shape[1], 50, 20, 1], 'epochs': 700, 'lr': 0.00015, 'momentum': 0.9},
    {'layers': [X_train_normalized.shape[1], 10, 10, 1], 'epochs': 800, 'lr': 0.0003, 'momentum': 0.85},
    {'layers': [X_train_normalized.shape[1], 100, 50, 10, 1], 'epochs': 500, 'lr': 0.0001, 'momentum': 0.9}
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

# Plot the scatter of the best-performing 3 set of parameters
best_results = sorted(results, key=lambda x: x['mse'])[:3]  # Sort by MSE, take the first 3 groups
plt.figure(figsize=(15, 5))

for i, result in enumerate(best_results):
    params = result['params']
    short_title = f"Layers: {params['layers']}, LR: {params['lr']}, Epochs: {params['epochs']}"  # Simplified Title
    y_pred = result['y_pred'] if 'y_pred' in result else model.predict(X_test_normalized) * y_std + y_mean
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.title(short_title)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")

plt.tight_layout()
plt.show()

# Drawing Training and Validation Error Curves for Optimal Hyperparametric Combinations
plt.figure(figsize=(15, 5))

for i, result in enumerate(best_results):
    params = result['params']
    short_title = f"Layers: {params['layers']}, LR: {params['lr']}, Epochs: {params['epochs']}"  # Simplified Title
    train_errors = result['train_errors'] if 'train_errors' in result else best_train_errors
    val_errors = result['val_errors'] if 'val_errors' in result else best_val_errors
    plt.subplot(1, 3, i + 1)
    plt.plot(train_errors, label="Train Error")
    plt.plot(val_errors, label="Validation Error")
    plt.title(short_title)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()

plt.tight_layout()
plt.show()






# Drawing Training and Validation Errors with epoch
if best_train_errors and best_val_errors:  # Ensure Error Data Exists
    plt.plot(best_train_errors, label='Train Error')
    plt.plot(best_val_errors, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Error')
    plt.legend()
    plt.show()

# Part 3.2 Model Results Comparison
# 1. multiple linear regression(MLR-F)
mlr = LinearRegression()
mlr.fit(X_train_normalized, y_train_normalized)
y_pred_mlr_normalized = mlr.predict(X_test_normalized)
y_pred_mlr = y_pred_mlr_normalized * y_std + y_mean
mse_mlr, mae_mlr, mape_mlr = evaluate_metrics(y_test, y_pred_mlr)

# 2. Neural Networks Using Open Source Libraries(BP-F)
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, learning_rate_init=0.001, random_state=42)
mlp.fit(X_train_normalized, y_train_normalized.ravel())
y_pred_mlp_normalized = mlp.predict(X_test_normalized)
y_pred_mlp = y_pred_mlp_normalized * y_std + y_mean
mse_mlp, mae_mlp, mape_mlp = evaluate_metrics(y_test, y_pred_mlp)

# Compare three models
print("\nModel Comparison:")
print(f"Neural Network - MSE: {test_error:.4f}, MAE: {mae_mlr:.4f}, MAPE: {mape_mlr:.2f}%")
print(f"Linear Regression - MSE: {mse_mlr:.4f}, MAE: {mae_mlr:.4f}, MAPE: {mape_mlr:.2f}%")
print(f"MLP Regressor - MSE: {mse_mlp:.4f}, MAE: {mae_mlp:.4f}, MAPE: {mape_mlp:.2f}%")

# Plotting Scatter Points
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Neural Network")
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
