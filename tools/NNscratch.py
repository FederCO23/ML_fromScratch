#
# NN-scratch: is a class to construct Neural Networks.
# The first version was created for a practice work with NN using a Jupyter Notebook.
# The intention is to build a library.
#
# author: Federico Pérez Bessi
#

import numpy as np
import time
import matplotlib.pyplot as plt
import warnings

class Neural_Network_3L:
    # 3L: three layers: input[0], hidden[1] and output[2] layers
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer):
        self.inputLayerSize = input_layer #2
        self.hiddenLayer1Size = hidden_layer1 #3
        self.hiddenLayer2Size = hidden_layer2  # ?
        self.outputLayerSize = output_layer #1
        self.initParams()

    def initParams(self):
        # Initialize the parameters matrices using smaller random values
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayer1Size) * 0.01
        # self.B1 = np.random.randn(1, self.hiddenLayerSize) #* 0.01
        self.B1 = np.zeros((1, self.hiddenLayer1Size))
        self.W2 = np.random.randn(self.hiddenLayer1Size, self.hiddenLayer2Size) * 0.01
        # self.B2 = np.random.randn(1, self.outputLayerSize) #* 0.01
        self.B2 = np.zeros((1, self.hiddenLayer2Size))
        self.W3 = np.random.randn(self.hiddenLayer2Size, self.outputLayerSize) * 0.01
        # self.B2 = np.random.randn(1, self.outputLayerSize) #* 0.01
        self.B3 = np.zeros((1, self.outputLayerSize))

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def relu(self,z):
        # Relu activation function
        return np.maximum(0,z)

    def forwardProp(self, X):
        # Propagate X through the network
        self.Z1 = np.matmul(X, self.W1) + self.B1  # equation (1)
        self.A1 = self.sigmoid(self.Z1)  # equation (2)
        self.Z2 = np.matmul(self.A1, self.W2) + self.B2  # equation (3)
        self.A2 = self.sigmoid(self.Z2)  # equation (4)
        self.Z3 = np.matmul(self.A2, self.W3) + self.B3  # added
        self.A3 = self.sigmoid(self.Z3)  # added

        return self.A3

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        m = X.shape[0]
        self.yHat = self.forwardProp(X)
        error = (self.yHat - y) ** 2
        J = np.sum(error) / (2 * m)
        return J

    def costFunctionPrime(self, X, y):
        self.yHat = self.forwardProp(X)
        m = X.shape[0]

        # dJdW2 and dJdB2 - from equation (6)
        delta3 = np.multiply((self.yHat - y), self.sigmoidPrime(self.Z3))
        dJdW3 = np.matmul(self.A2.T, delta3)
        dJdB3 = delta3

        # dJdW2 and dJdB2 - from equation (6)
        delta2 = np.matmul(delta3, self.W3.T) * self.sigmoidPrime(self.Z2)
        dJdW2 = np.matmul(self.A1.T, delta2)
        dJdB2 = delta2

        # dJdW1 and dJdB1 - from equation (7)
        delta1 = np.matmul(delta2, self.W2.T) * self.sigmoidPrime(self.Z1)
        dJdW1 = np.matmul(X.T, delta1)
        dJdB1 = delta1

        dJdW3 = dJdW3 / m
        dJdW2 = dJdW2 / m
        dJdW1 = dJdW1 / m
        dJdB3 = np.sum(dJdB3, axis=0, keepdims=True) / m
        dJdB2 = np.sum(dJdB2, axis=0, keepdims=True) / m
        dJdB1 = np.sum(dJdB1, axis=0, keepdims=True) / m

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3

    def sigmoidPrime(self, z):
        # Sigmoid activation function derivative
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def reluPrime(self,z):
        # Relu activation function derivative
        return np.where(z > 0, 1, 0)

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1,W2,B2 and B1 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel(), self.B1.ravel(), self.B2.ravel(), self.B3.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayer1Size * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayer1Size))
        W2_end = W1_end + self.hiddenLayer2Size * self.hiddenLayer1Size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))
        W3_end = W2_end + self.hiddenLayer2Size * self.outputLayerSize
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayer2Size, self.outputLayerSize))

        B1_end = W3_end + self.hiddenLayer1Size
        self.B1 = np.reshape(params[W3_end:B1_end], (1, self.hiddenLayer1Size))
        B2_end = B1_end + self.hiddenLayer2Size
        self.B2 = np.reshape(params[B1_end:B2_end], (1, self.hiddenLayer2Size))
        B3_end = B2_end + self.outputLayerSize
        self.B3 = np.reshape(params[B2_end:B3_end], (1, self.outputLayerSize))

    def numgrad2Devs(self, numgrad):
        # converts the numerical gradient array to param matrix derivatives
        dJdW1_start = 0
        dJdW1_end = self.hiddenLayer1Size * self.inputLayerSize
        dJdW1 = np.reshape(numgrad[dJdW1_start:dJdW1_end], (self.inputLayerSize, self.hiddenLayer1Size))
        dJdW2_end = dJdW1_end + self.hiddenLayer1Size * self.hiddenLayer2Size
        dJdW2 = np.reshape(numgrad[dJdW1_end:dJdW2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))
        dJdW3_end = dJdW2_end + self.hiddenLayer2Size * self.outputLayerSize
        dJdW3 = np.reshape(numgrad[dJdW2_end:dJdW3_end], (self.hiddenLayer2Size, self.outputLayerSize))

        dJdB1_end = dJdW3_end + self.hiddenLayer1Size
        dJdB1 = np.reshape(numgrad[dJdW3_end:dJdB1_end], (1, self.hiddenLayer1Size))
        dJdB2_end = dJdB1_end + self.hiddenLayer2Size
        dJdB2 = np.reshape(numgrad[dJdB1_end:dJdB2_end], (1, self.hiddenLayer2Size))
        dJdB3_end = dJdB2_end + self.outputLayerSize
        dJdB3 = np.reshape(numgrad[dJdB2_end:dJdB3_end], (1, self.outputLayerSize))

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3

    def computeNumericalGradient(self, X, y):
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costFunction(X, y)

            self.setParams(paramsInitial - perturb)
            loss1 = self.costFunction(X, y)

            # added code to avoid errors: ensure loss1 and loss2 are scalars
            loss2 = loss2.item() if np.ndim(loss2) > 0 else loss2
            loss1 = loss1.item() if np.ndim(loss1) > 0 else loss1

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)

            # Return the value we changed to zero:
            perturb[p] = 0

        # Return Params to original value:
        self.setParams(paramsInitial)

        return numgrad

    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel(), dJdB1.ravel(), dJdB2.ravel(), dJdB3.ravel()))

    def train_gradientDescent(self, X, y, alpha, epochs, init=True, verbose=True, debug=True, numerical=False,
                              logtime=False):

        # Initialize the parameters
        if init:
            self.initParams()

        J_hist = []
        param_hist = []

        # Record start time
        start_time = time.time()

        # Gradient Descent iterations
        for i in range(epochs):

            # Calculate the Cost and log it
            cost = self.costFunction(X, y)
            J_hist.append(cost.item())

            # Calculates the derivatives
            if numerical:
                numgrad = self.computeNumericalGradient(X, y)
                dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.numgrad2Devs(numgrad)
            else:
                dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.costFunctionPrime(X, y)

            if verbose:
                # Print intermediate values
                if i % int(epochs / 10) == 0:
                    # Print the result
                    print(f'Epoch {i}: Cost = {cost}')

            # Update parameters
            self.W1 = self.W1 - alpha * dJdW1
            self.W2 = self.W2 - alpha * dJdW2
            self.W3 = self.W3 - alpha * dJdW3
            self.B1 = self.B1 - alpha * dJdB1
            self.B2 = self.B2 - alpha * dJdB2
            self.B3 = self.B3 - alpha * dJdB3

            if debug:
                params = self.getParams()
                param_hist.append(params)

        if logtime:
            # Record end time
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f'Elapsed time: {elapsed_time:2.1f} s')

        return J_hist, np.array(param_hist)


class Neural_Network_3L_LogR(Neural_Network_3L):
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer):
        super().__init__(input_layer, hidden_layer1, hidden_layer2, output_layer)

    # Add a new method to predict the binary output
    def predict(self, X):
        A3 = self.forwardProp(X)
        predictions = (A3 > 0.5).astype(int)
        return predictions

    # Add a new Cost J function using Binary Cross-Entropy
    def costFunction(self, X, y):
        m = X.shape[0]
        self.yHat = self.forwardProp(X)
        cost = -np.sum(y * np.log(self.yHat) + (1 - y) * np.log(1 - self.yHat)) / m
        # Squeeze the cost to remove single-dimensional entries
        cost = np.squeeze(cost)
        return cost

    def costFunctionPrime(self, X, y):
        m = X.shape[0]

        # initialize self.yHat, self.A1, self.A2, self.Z1 and self.Z2
        self.yHat = self.forwardProp(X)

        # Parameters and intermediates derivatives
        dJdZ3 = self.yHat - y
        dJdW3 = np.matmul(self.A2.T, dJdZ3)
        dJdB3 = dJdZ3

        dJdZ2 = np.multiply(np.matmul(dJdZ3, self.W3.T), self.sigmoidPrime(self.Z2))  # eq (5)
        dJdW2 = np.matmul(self.A1.T, dJdZ2)  # eq (3)
        dJdB2 = dJdZ2  # eq (4)

        dJdZ1 = np.multiply(np.matmul(dJdZ2, self.W2.T), self.sigmoidPrime(self.Z1))  # eq (5)
        dJdW1 = np.matmul(X.T, dJdZ1)  # eq (3)
        dJdB1 = dJdZ1  # eq (4)

        dJdW3 = dJdW3 / m
        dJdB3 = np.sum(dJdB3, axis=0, keepdims=True) / m
        dJdW2 = dJdW2 / m
        dJdB2 = np.sum(dJdB2, axis=0, keepdims=True) / m
        dJdW1 = dJdW1 / m
        dJdB1 = np.sum(dJdB1, axis=0, keepdims=True) / m

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3


class Neural_Network_3L_LogR_SM:
    # SM stands for SoftMax.
    # Architecture:
    #    - 2x Hidden layers with RELU activation functions
    #    - 1x Output layer with LINEAR activation functions
    # Methods are adapted as following.

    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer):
        self.inputLayerSize = input_layer
        self.hiddenLayer1Size = hidden_layer1
        self.hiddenLayer2Size = hidden_layer2
        self.outputLayerSize = output_layer
        self.initParams()

    def initParams(self):
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayer1Size) * 0.01
        self.B1 = np.zeros((1, self.hiddenLayer1Size))
        self.W2 = np.random.randn(self.hiddenLayer1Size, self.hiddenLayer2Size) * 0.01
        self.B2 = np.zeros((1, self.hiddenLayer2Size))
        self.W3 = np.random.randn(self.hiddenLayer2Size, self.outputLayerSize) * 0.01
        self.B3 = np.zeros((1, self.outputLayerSize))

    def relu(self,z):
        # Relu activation function
        return np.maximum(0,z)


    def reluPrime(self, z):
        # Relu activation function derivative
        return np.where(z > 0, 1, 0)


    def forwardProp(self, X):
        # Propagate X through the network
        self.Z1 = np.matmul(X, self.W1) + self.B1  # equation (1)
        self.A1 = self.relu(self.Z1)  # equation (2)
        self.Z2 = np.matmul(self.A1, self.W2) + self.B2  # equation (3)
        self.A2 = self.relu(self.Z2)  # equation (4)
        self.Z3 = np.matmul(self.A2, self.W3) + self.B3
        self.A3 = self.Z3  # Linear activation for the output layer

        return self.A3

    def predict(self, X):
        A3 = self.forwardProp(X)
        exp_scores = np.exp(np.clip(A3 - np.max(A3, axis=1, keepdims=True), -500, 500))  # For numerical stability
        predictions = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return predictions

    def costFunctionPrime(self, X, y):
        m = X.shape[0]
        self.yHat = self.predict(X)  # Output after softmax

        # Gradient of the loss with respect to Z3 (input to softmax layer)
        dJdZ3 = self.yHat
        dJdZ3[np.arange(m), y] -= 1
        dJdZ3 = dJdZ3 / m

        # Gradient of the cost with respect to W3 and B3
        dJdW3 = np.dot(self.A2.T, dJdZ3)
        dJdB3 = np.sum(dJdZ3, axis=0, keepdims=True)

        # Backpropagation to hidden layer 2
        dJdZ2 = np.dot(dJdZ3, self.W3.T) * self.reluPrime(self.Z2)
        dJdW2 = np.dot(self.A1.T, dJdZ2)
        dJdB2 = np.sum(dJdZ2, axis=0, keepdims=True)

        # Backpropagation to hidden layer 1
        dJdZ1 = np.dot(dJdZ2, self.W2.T) * self.reluPrime(self.Z1)
        dJdW1 = np.dot(X.T, dJdZ1)
        dJdB1 = np.sum(dJdZ1, axis=0, keepdims=True)

        # Clip gradients to avoid exploding gradients
        clip_value = 5.0
        dJdW3 = np.clip(dJdW3, -clip_value, clip_value)
        dJdB3 = np.clip(dJdB3, -clip_value, clip_value)
        dJdW2 = np.clip(dJdW2, -clip_value, clip_value)
        dJdB2 = np.clip(dJdB2, -clip_value, clip_value)
        dJdW1 = np.clip(dJdW1, -clip_value, clip_value)
        dJdB1 = np.clip(dJdB1, -clip_value, clip_value)

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3



    def costFunction(self, X, y):
        m = X.shape[0]
        self.yHat = self.predict(X)

        # adding epsilon to avoid /0
        epsilon = 1e-10
        log_likelihoods = -np.log(self.yHat[np.arange(m), y] + epsilon)

        # Compute the sparse categorical cross-entropy cost function
        cost = np.sum(log_likelihoods) / m

        return cost

    def train_gradientDescent(self, X, y, alpha, epochs, init=True, verbose=True, debug=True, numerical=False,
                              logtime=False, batch_size=32):
        def data_generator(X, y, batch_size):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                yield X_shuffled[start:end], y_shuffled[start:end]

        # Initialize the parameters
        if init:
            self.initParams()

        J_hist = []
        param_hist = []

        # Record start time
        start_time = time.time()

        # Gradient Descent iterations
        for i in range(epochs):

            nb_batch = int(np.ceil(X.shape[0] / batch_size))
            generator = data_generator(X, y, batch_size)

            for _ in range(nb_batch):

                # fill the batch
                X_batch, y_batch = next(generator)


                # Calculates the derivatives
                if numerical:
                    numgrad = self.computeNumericalGradient(X_batch, y_batch)
                    dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.numgrad2Devs(numgrad)
                else:
                    dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.costFunctionPrime(X_batch, y_batch)

                # Update parameters
                self.W1 = self.W1 - alpha * dJdW1
                self.W2 = self.W2 - alpha * dJdW2
                self.W3 = self.W3 - alpha * dJdW3
                self.B1 = self.B1 - alpha * dJdB1
                self.B2 = self.B2 - alpha * dJdB2
                self.B3 = self.B3 - alpha * dJdB3

                if debug:
                    params = self.getParams()
                    param_hist.append(params)

            # Calculate the Cost and log it
            cost = self.costFunction(X_batch, y_batch)
            J_hist.append(cost.item())

            if verbose:
                # Print intermediate values
                if i % int(epochs / 10) == 0:
                    # Print the result
                    print(f'Epoch {i}: Cost = {cost}')

        if logtime:
            # Record end time
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f'Elapsed time: {elapsed_time:2.1f} s')

        return J_hist, np.array(param_hist)


    def getParams(self):
        # Get W1,W2,B2 and B1 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel(), self.B1.ravel(), self.B2.ravel(), self.B3.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayer1Size * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayer1Size))
        W2_end = W1_end + self.hiddenLayer2Size * self.hiddenLayer1Size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))
        W3_end = W2_end + self.hiddenLayer2Size * self.outputLayerSize
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayer2Size, self.outputLayerSize))

        B1_end = W3_end + self.hiddenLayer1Size
        self.B1 = np.reshape(params[W3_end:B1_end], (1, self.hiddenLayer1Size))
        B2_end = B1_end + self.hiddenLayer2Size
        self.B2 = np.reshape(params[B1_end:B2_end], (1, self.hiddenLayer2Size))
        B3_end = B2_end + self.outputLayerSize
        self.B3 = np.reshape(params[B2_end:B3_end], (1, self.outputLayerSize))

    def numgrad2Devs(self, numgrad):
        # converts the numerical gradient array to param matrix derivatives
        dJdW1_start = 0
        dJdW1_end = self.hiddenLayer1Size * self.inputLayerSize
        dJdW1 = np.reshape(numgrad[dJdW1_start:dJdW1_end], (self.inputLayerSize, self.hiddenLayer1Size))
        dJdW2_end = dJdW1_end + self.hiddenLayer1Size * self.hiddenLayer2Size
        dJdW2 = np.reshape(numgrad[dJdW1_end:dJdW2_end], (self.hiddenLayer1Size, self.hiddenLayer2Size))
        dJdW3_end = dJdW2_end + self.hiddenLayer2Size * self.outputLayerSize
        dJdW3 = np.reshape(numgrad[dJdW2_end:dJdW3_end], (self.hiddenLayer2Size, self.outputLayerSize))

        dJdB1_end = dJdW3_end + self.hiddenLayer1Size
        dJdB1 = np.reshape(numgrad[dJdW3_end:dJdB1_end], (1, self.hiddenLayer1Size))
        dJdB2_end = dJdB1_end + self.hiddenLayer2Size
        dJdB2 = np.reshape(numgrad[dJdB1_end:dJdB2_end], (1, self.hiddenLayer2Size))
        dJdB3_end = dJdB2_end + self.outputLayerSize
        dJdB3 = np.reshape(numgrad[dJdB2_end:dJdB3_end], (1, self.outputLayerSize))

        return dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3

    def computeNumericalGradient(self, X, y):
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costFunction(X, y)

            self.setParams(paramsInitial - perturb)
            loss1 = self.costFunction(X, y)

            # added code to avoid errors: ensure loss1 and loss2 are scalars
            loss2 = loss2.item() if np.ndim(loss2) > 0 else loss2
            loss1 = loss1.item() if np.ndim(loss1) > 0 else loss1

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)

            # Return the value we changed to zero:
            perturb[p] = 0

        # Return Params to original value:
        self.setParams(paramsInitial)

        return numgrad

    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3, dJdB1, dJdB2, dJdB3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel(), dJdB1.ravel(), dJdB2.ravel(), dJdB3.ravel()))







def oneHotarray(Y):
    b = np.zeros((Y.size, Y.max() + 1))
    b[np.arange(Y.size), Y.T] = 1
    return b.T

# adapt the plot labeling, it has changed
def plotParams(param_hist):
    labels = ['W1_11', 'W1_12', 'W1_13', 'W1_21', 'W1_22', 'W1_23', 'W2_11', 'W2_12', 'W2_13', 'B1_11', 'B1_12',
              'B1_13', 'B2_11']
    steps_range = np.arange(param_hist.shape[0])
    # steps_range = np.arange(20)
    plt.figure(figsize=(10, 8))
    for i in range(param_hist.shape[1]):
        # for i in range(20):
        plt.plot(steps_range, param_hist[:, i], label=f'Column {labels[i]}')
        mid_point = len(steps_range) // 2
        plt.text(float(steps_range[mid_point]), param_hist[mid_point, i], f'{labels[i]}', fontsize=8, color='blue')

    # Add labels and title
    plt.xlabel('epochs')
    plt.ylabel('Param. Value')
    plt.title('Parameters change during Gradient Descent computing')

    # Show plot
    plt.show()


def plotJhist(J_hist):
    steps_range = np.arange(len(J_hist))
    plt.figure(figsize=(6, 3))
    plt.plot(steps_range, J_hist[:])
    plt.grid(True)
    # Add labels and title
    plt.xlabel('epochs')
    plt.ylabel('Cost J')
    plt.title('Cost (J) change during Gradient Descent')
    plt.show()


def scaleX2F(X):
    X1 = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X2 = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    Xn = np.stack((X1, X2), axis=1)
    return Xn

def load_data(bin=True):
    X = np.load("../data/X.npy")
    y = np.load("../data/y.npy")
    if bin:
        X = X[0:1000]
        y = y[0:1000]
    return X, y


def plot_inputs(x,y):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = x.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(top=0.9)  # Adjust top to provide more space for titles

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = x[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
    fig.suptitle("Labels for Input images", fontsize=16)
    plt.show()

