import numpy as np #for linear algebra
import pandas as pd #to read data
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/emnist/emnist-balanced-train.csv')

output_node = 47 #no. of output classes

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
    # Input: 784, Hidden1: 128, Hidden2: 64, Output: 47
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    
    # New Hidden Layer 2
    W2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    
    # Output Layer (Connecting Hidden2 to Output)
    W3 = np.random.rand(47, 64) - 0.5
    b3 = np.random.rand(47, 1) - 0.5
    
    return W1, b1, W2, b2, W3, b3

def leaky_ReLU(Z, alpha=0.01):
    return np.maximum(alpha * Z, Z)

def d_leaky_ReLU(Z, alpha=0.01):
    dZ = np.ones_like(Z)
    dZ[Z < 0] = alpha
    return dZ

def softmax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = X @ W1 + b1
    A1 = leaky_ReLU(0, Z1)  # ReLU activation

    Z2 = W2 @ A1 + b2
    A2 = leaky_ReLU(0, Z2)  # ReLU activation

    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)  # Softmax activation

    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    m = Y.size

    dZ3 = A3 - one_hot_Y
    dW3 = (1/m) * (A2.T @ dZ3)
    db3 = (1/m) * np.sum(dZ3)

    dZ2 = W3.T @ dZ3 * d_leaky_ReLU(Z2)
    dW2 = (1/m) * (A1.T @ dZ2)
    db2 = (1/m) * np.sum(dZ2)

    dZ1 = W2.T @ dZ2 * d_leaky_ReLU(Z1)
    dW1 = (1/m) * (X.T @ dZ1)
    db1 = (1/m) * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3    
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, epochs):
    W1, b1, W2, b2, W3, b3 = init_params()
    costs = []
    
    for i in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3,
                                               dW1, db1, dW2, db2, dW3, db3,
                                               alpha)
        
        if i % 10 == 0:
            print("Epoch:", i)
            print("Accuracy:", get_accuracy(get_predictions(A3), Y))
    
    return W1, b1, W2, b2, W3, b3, costs

W1, b1, W2, b2, W3, b3, costs = gradient_descent(X_train.T, Y_train, 0.1, 100) 



def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
