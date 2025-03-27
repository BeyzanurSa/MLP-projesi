import numpy as np

class MLPClassifier:
    def __init__(self, n_x, n_h1, n_h2, n_y=1, learning_rate=0.01):
        self.n_x = n_x
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.n_y = n_y
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters()
        self.costs = []  # Maliyetleri kaydetmek için liste
    
    def initialize_parameters(self):
        np.random.seed(42)
        W1 = np.random.randn(self.n_h1, self.n_x) * np.sqrt(1 / self.n_x)  # Xavier Initialization
        b1 = np.zeros((self.n_h1, 1))
        W2 = np.random.randn(self.n_h2, self.n_h1) * np.sqrt(1 / self.n_h1)
        b2 = np.zeros((self.n_h2, 1))
        W3 = np.random.randn(self.n_y, self.n_h2) * np.sqrt(1 / self.n_h2)
        b3 = np.zeros((self.n_y, 1))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward_propagation(self, X):
        W1, b1, W2, b2, W3, b3 = self.parameters.values()
        Z1 = np.dot(W1, X.T) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = self.sigmoid(Z3)
        return A3, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    
    def compute_cost(self, A3, Y):
        m = A3.shape[1]
        cost = -(np.dot(np.log(A3), Y) + np.dot(np.log(1 - A3), (1 - Y))) / m
        return float(np.squeeze(cost))
    
    def backpropagation(self, X, Y, cache):
        m = X.shape[0]
        W1, W2, W3 = self.parameters["W1"], self.parameters["W2"], self.parameters["W3"]
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
        return {"dW1": dW1, "dW2": dW2, "dW3": dW3, "db1": db1, "db2": db2, "db3": db3}
    
    def update_parameters(self, grads):
        for key in grads:
            self.parameters[key[1:]] -= self.learning_rate * grads[key]
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, n_steps=5000, print_cost=True, optimizer=None):
        if optimizer is not None:
            print(f"Optimizer '{optimizer}' is specified but not used in this implementation.")
        self.train_costs = []
        self.test_costs = []
        for i in range(n_steps):
            # Forward propagation
            A3_train, cache = self.forward_propagation(X_train)
            cost = self.compute_cost(A3_train, y_train)
            grads = self.backpropagation(X_train, y_train, cache)
            self.update_parameters(grads)
            self.train_costs.append(cost)

            # Compute validation cost if X_test and y_test are provided
            if X_test is not None and y_test is not None:
                A3_test, _ = self.forward_propagation(X_test)
                test_cost = self.compute_cost(A3_test, y_test)
                self.test_costs.append(test_cost)

            # Print cost every 1000 steps
            if print_cost and i % 1000 == 0:
                print(f"Step {i}: Training Cost = {cost:.6f}")
                if X_test is not None and y_test is not None:
                    print(f"Step {i}: Validation Cost = {test_cost:.6f}")
    
    def predict(self, X):
        A3, _ = self.forward_propagation(X)
        return A3 > 0.5
