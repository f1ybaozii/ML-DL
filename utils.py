import numpy as np
import matplotlib.pyplot as plt


def paint(x: np.ndarray,y: np.ndarray):
    '''Paint the scatter plot of x and y'''
    plt.figure()
    plt.scatter(x,y)
    plt.show()


def myfit(model,x: np.ndarray,y: np.ndarray):
    '''Fit the model to the data
    Args:
        model: the model to fit
        x: the input data
        y: the output data
    '''
    model.fit(x,y)

def split_data(x: np.ndarray,y: np.ndarray,split_ratio: float,shuffle: bool = True):
    '''Split the data into training and testing data
    Args:
        x: the input data
        y: the output data
        split_ratio: the ratio of the training data
        shuffle: whether to shuffle the data before splitting
    Returns:
        x_train: the training input data
        y_train: the training output data
        x_test: the testing input data
        y_test: the testing output data
    '''
    if shuffle:
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    split_idx = int(len(x) * split_ratio)
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]
    return x_train,y_train,x_test,y_test

def display(x_test, y_test, regressor):
    plt.figure()

    xx = np.linspace(np.min(x_test), np.max(x_test), 1000)
    yy = regressor.predict(xx[:, np.newaxis])

    plt.scatter(x_test, y_test)
    plt.plot(xx, yy, c='red')
    plt.show()

def sigmoid(x):
    '''The sigmoid function'''
    return np.exp(x) / (1 + np.exp(x))

def relu(x):
    '''The ReLU function'''
    return np.maximum(0,x)

