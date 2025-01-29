import numpy as np
import math
import time
import matplotlib.pyplot as plt

# -------- activation functions -------
class Relu():
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x, grad):
        return grad * (x > 0)
    
class Identity():
    def __init__(self):
        pass

    def forward(self, x):
        return x
    
    def backward(self, x, grad):
        return grad
    
class LeakyRelu():
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)
    
    def backward(self, x, grad):
        return grad * (x > 0) + self.alpha * grad * (x <= 0)

# identity = lambda z: z

# identity_back = lambda xbar, z: xbar
# -------------------------------------------


# ---------- initialization -----------
# def glorot(nin, nout):
#    # TODO
#     return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    # y.reshape(yhat.shape)
    # assert yhat.shape == y.shape
    return np.mean((yhat - y) ** 2)

def mse_back(yhat, y):
    # assert yhat.shape == y.shape
    return 2 * (yhat - y) / yhat.shape[1]

# -----------------------------------


# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=Identity(), alpha=1e-4):
        self.alpha = alpha
        # TODO: initialize and setup variables
        self.weights = np.random.randn(nout, nin) * np.sqrt(2/nin)
        self.bias = np.zeros((nout,1))
        self.activation = activation

        # if activation == relu:
        #     self.activation_back = relu_back
        # if activation == identity:
        #     self.activation_back = identity_back

        # initialize cache
        self.activation = activation
        self.cache = {}

    def forward(self, X, train=True):
        z = np.dot(self.weights,X)  + self.bias
        Xnew = self.activation.forward(z)
        if train:
            self.cache['X'] = X
            self.cache['Z'] = z
            self.cache['Xnew'] = Xnew
        return Xnew

    def backward(self, Xnewbar):
        dZ = self.activation.backward(self.cache['Z'], Xnewbar)
        self.dW = np.dot(dZ, self.cache['X'].T) / self.cache['X'].shape[1]
        self.db = np.mean(dZ, axis=1, keepdims=True) / self.cache['X'].shape[1]
        Xbar = np.dot(self.weights.T, dZ)
        return Xbar
    
    def step(self):
        self.weights -= self.dW * self.alpha
        self.bias -= self.db * self.alpha


class Network:

    def __init__(self, layers, loss, input_size, alpha=1e-4):
        # initialization
        self.layers = [None] * len(layers)
        self.loss = loss
        for i in range(len(layers)):
            if i == 0:
                self.layers[i] = Layer(input_size, layers[i], LeakyRelu(), alpha)
            else:
                self.layers[i] = Layer(layers[i-1], layers[i], LeakyRelu(), alpha)
        self.layers.append(Layer(layers[-1], 1, Identity(), alpha))
        self.Xs = [None] * (len(layers) + 1)

        if loss == mse:
            self.loss_back = mse_back

    def forward(self, X, y, train=True):
        """ 
        args: X: input, y: target, train: True if training, False if testing
        output: L: loss, yhat: prediction
        """
        for i, layer in enumerate(self.layers):
            X = layer.forward(X, train)
            if train:
                self.Xs[i] = X
        yhat = X
        L = self.loss(yhat, y)  
        self.y = y
        return L, yhat

    def backward(self):
        Xbar = self.loss_back(self.Xs[-1], self.y)
        Xbar = self.layers[-1].backward(Xbar)
        self.layers[-1].step()
        for i in range(len(self.layers)-2,-1,-1):
            Xbar = self.layers[i].backward(Xbar)
            self.layers[i].step()
        return Xbar

    def setAlpha(self, alpha):
        for layer in self.layers:
            layer.alpha = alpha

if __name__ == '__main__':

    # ---------- data preparation ----------------
    # Initialize lists for the numeric data and the string data
    numeric_data = []

    # Read the text file
    with open('auto-mpg.data', 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()

            # Check if any of the first 8 columns contain '?'
            if '?' in columns[:8]:
                continue  # Skip this line if there's a missing value

            # Convert the first 8 columns to floats and append to numeric_data
            numeric_data.append([float(value) for value in columns[:8]])

    # Convert numeric_data to a numpy array for easier manipulation
    numeric_array = np.array(numeric_data)

    # Shuffle the numeric array and the corresponding string array
    nrows = numeric_array.shape[0]
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    shuffled_numeric_array = numeric_array[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    train_numeric = shuffled_numeric_array[:split_index]
    test_numeric = shuffled_numeric_array[split_index:]

    # separate inputs/outputs
    Xtrain = train_numeric[:, 1:]
    ytrain = train_numeric[:, 0]

    Xtest = test_numeric[:, 1:]
    ytest = test_numeric[:, 0]

    # normalize
    Xmean = np.mean(Xtrain, axis=0)
    Xstd = np.std(Xtrain, axis=0)
    ymean = np.mean(ytrain)
    ystd = np.std(ytrain)

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    # reshape arrays (opposite order of pytorch, here we have nx x ns).
    # I found that to be more conveient with the way I did the math operations, but feel free to setup
    # however you like.
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    ytrain = np.reshape(ytrain, (1, len(ytrain)))
    ytest = np.reshape(ytest, (1, len(ytest)))

    # ------------------------------------------------------------

    # l1 = Layer(7, ?, relu)
    # # TODO
    # layers = [l1, l2, l3]
    # network = Network(layers, mse)
    # alpha = ?
    # optimizer = GradientDescent(alpha)
    network = Network([1500,1500],mse,7,1e-2)

    train_losses = []
    test_losses = []
    epochs = 749
    for i in range(epochs):
        # TODO: run train set, backprop, step
        L, yhat = network.forward(Xtrain, ytrain)
        train_losses.append(L)
        network.backward()

        # TODO: run test set
        L_test, yhat = network.forward(Xtest, ytest, train=False)
        # temp = abs(L_test - L)/L
        if i % 20 == 0 and i != 0:
            curr_alpha = network.layers[0].alpha
            network.setAlpha(curr_alpha/4)
        if i % 150 == 0:
            network.setAlpha(1e-1)
        test_losses.append(L_test)
        if i % 25 == 0:
            print("epoch: ", i)
        L_prior = L

    # --- inference ----
    

    # unnormalize
    yhat = (yhat * ystd) + ymean
    ytest = (ytest * ystd) + ymean

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()


    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.plot([10, 45], [10, 45], "--")

    print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

    plt.show()
    time.sleep(5)