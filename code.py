import numpy as np #for linear algebra
import pandas as pd #to read data
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/emnist-balanced-train/emnist-balanced-train.csv')

data.head()

data  = np.array(data)

m, n = data.shape
#m is no. of rows/ examples
#n is no. of columns/ features + 1
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


def init_params():
    #initialize weights and biases for 3-layer neural network
    H1 = 128
    H2 = 64
    W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    b1 = np.random.randn(10, 1)

    W2 = np.random.randn(H1, H2) * np.sqrt(2 / H1)
    b2 = np.zeros(H2)

    W3 = np.random.randn(H2, 47) * np.sqrt(2 / H2)
    b3 = np.zeros(47)

    return W1, b1, W2, b2, W3, b3

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def d_leaky_relu(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation

    Z2 = W2.dot(A1) + b2
    A2 = np.maximum(0, Z2)  # ReLU activation

    Z3 = W3.dot(A2) + b3
    exp_scores = np.exp(Z3 - np.max(Z3, axis=0, keepdims=True))
    A3 = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)  # Softmax activation

    return Z1, A1, Z2, A2, Z3, A3

def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A