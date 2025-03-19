import numpy as np

class MLPClassifier:
    def __init__(self, n_x, n_h, n_h2=None, n_y=1, learning_rate=0.01, activation_function='tanh'):
        self.n_x = n_x
        self.n_h = n_h
        self.n_h2 = n_h2  # Second hidden layer (optional)
        self.n_y = n_y
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.parameters = self.initialize_parameters()
        self.costs = []  # To store the cost during training

    def initialize_parameters(self):
        np.random.seed(42)
        W1 = np.random.randn(self.n_h, self.n_x) * np.sqrt(1 / self.n_x)  # Xavier Initialization
        b1 = np.zeros((self.n_h, 1))
        if self.n_h2:  # If a second hidden layer is specified
            W2 = np.random.randn(self.n_h2, self.n_h) * np.sqrt(1 / self.n_h)
            b2 = np.zeros((self.n_h2, 1))
            W3 = np.random.randn(self.n_y, self.n_h2) * np.sqrt(1 / self.n_h2)
            b3 = np.zeros((self.n_y, 1))
            return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        else:  # For a 2-layer network
            W2 = np.random.randn(self.n_y, self.n_h) * np.sqrt(1 / self.n_h)
            b2 = np.zeros((self.n_y, 1))
            return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward_propagation(self, X):
        W1, b1 = self.parameters["W1"], self.parameters["b1"]
        Z1 = np.dot(W1, X.T) + b1
        A1 = self.relu(Z1) if self.activation_function == 'relu' else np.tanh(Z1)

        if self.n_h2:  # For a 3-layer network
            W2, b2 = self.parameters["W2"], self.parameters["b2"]
            Z2 = np.dot(W2, A1) + b2
            A2 = self.relu(Z2) if self.activation_function == 'relu' else np.tanh(Z2)

            W3, b3 = self.parameters["W3"], self.parameters["b3"]
            Z3 = np.dot(W3, A2) + b3
            A3 = self.sigmoid(Z3)
            return A3, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
        else:  # For a 2-layer network
            W2, b2 = self.parameters["W2"], self.parameters["b2"]
            Z2 = np.dot(W2, A1) + b2
            A2 = self.sigmoid(Z2)
            return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def compute_cost(self, A, Y):
        m = Y.shape[0]
        cost = -(np.dot(np.log(A), Y) + np.dot(np.log(1 - A), (1 - Y))) / m
        return float(np.squeeze(cost))

    def backpropagation(self, X, Y, cache):
        m = X.shape[0]
        W1 = self.parameters["W1"]
        if self.n_h2:  # For a 3-layer network
            W2, W3 = self.parameters["W2"], self.parameters["W3"]
            A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]

            dZ3 = A3 - Y.T
            dW3 = np.dot(dZ3, A2.T) / m
            db3 = np.sum(dZ3, axis=1, keepdims=True) / m

            dZ2 = np.dot(W3.T, dZ3) * (1 - np.power(A2, 2))
            dW2 = np.dot(dZ2, A1.T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m

            dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
            dW1 = np.dot(dZ1, X) / m
            db1 = np.sum(dZ1, axis=1, keepdims=True) / m

            return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
        else:  # For a 2-layer network
            W2 = self.parameters["W2"]
            A1, A2 = cache["A1"], cache["A2"]

            dZ2 = A2 - Y.T
            dW2 = np.dot(dZ2, A1.T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m

            dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
            dW1 = np.dot(dZ1, X) / m
            db1 = np.sum(dZ1, axis=1, keepdims=True) / m

            return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads):
        for key in grads:
            self.parameters[key[1:]] -= self.learning_rate * grads[key]

    def fit(self, X_train, y_train, n_steps=5000, print_cost=True):
        for i in range(n_steps):
            A, cache = self.forward_propagation(X_train)
            cost = self.compute_cost(A, y_train)
            grads = self.backpropagation(X_train, y_train, cache)
            self.update_parameters(grads)
            self.costs.append(cost)
            if print_cost and i % 1000 == 0:
                print(f"Step {i}: Cost = {cost:.6f}")

    def predict(self, X):
        A, _ = self.forward_propagation(X)
        return A > 0.5
