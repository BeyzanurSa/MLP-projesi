import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from models.mlp_2layer import MLPClassifier
from models.mlp_3layer import MLPClassifier as M3LPClassifier
from experiments.results_analysis import evaluate_model, plot_confusion_matrix, plot_learning_curve, save_results
from sklearn.neural_network import MLPClassifier as SklearnMLP
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim

# Set random seed for all libraries to ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Veri yükleme ve ayırma
df = pd.read_csv("MLP/data/BankNote_Authentication.csv").sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy().reshape(-1, 1)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameters for all models
learning_rate = 0.01
n_hidden = 6
max_iter = 5000
batch_size = 32  # If using mini-batch SGD

# Model eğitimi ve accuracy takibi
def train_and_evaluate(model, model_name):
    # Ensure SGD optimizer is used
    model.fit(X_train, y_train, X_test, y_test, n_steps=max_iter, optimizer='sgd')
    
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()
    y_train_true = y_train.flatten()
    y_test_true = y_test.flatten()
    
    acc_test = np.mean(y_pred_test == y_test_true) * 100
    print(f"{model_name} Test Accuracy: {acc_test:.2f}%")
    
    return acc_test

# 2-Layer MLP
mlp_2layer = MLPClassifier(n_x=X_train.shape[1], n_h=n_hidden)
acc_2layer = train_and_evaluate(mlp_2layer, "2-Layer MLP")

# 3-Layer MLP
mlp_3layer = M3LPClassifier(n_x=X_train.shape[1], n_h1=n_hidden, n_h2=n_hidden)
acc_3layer = train_and_evaluate(mlp_3layer, "3-Layer MLP")

# En iyi modeli seçme
best_model = "2-Layer MLP" if acc_2layer > acc_3layer else "3-Layer MLP"
print(f"Seçilen en iyi model: {best_model}")

# Scikit-learn MLP Modeli - Using SGD optimizer
sklearn_mlp = SklearnMLP(hidden_layer_sizes=(n_hidden, n_hidden), 
                         activation='relu', 
                         solver='sgd',  # Changed to SGD
                         learning_rate_init=learning_rate,
                         max_iter=max_iter, 
                         random_state=42)
sklearn_mlp.fit(X_train, y_train.ravel())
sklearn_acc = sklearn_mlp.score(X_test, y_test) * 100
print(f"Scikit-learn MLP Test Accuracy: {sklearn_acc:.2f}%")

# PyTorch Modeli Tanımlama
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights consistently
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# PyTorch Modelini Eğitme
input_size = X_train.shape[1]
model = PyTorchMLP(input_size, n_hidden, 1)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Changed to SGD

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).squeeze()  # Ensure target tensor is 1D
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).squeeze()    # Ensure target tensor is 1D

# Training loop
for epoch in range(max_iter):
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()  # Ensure output is 1D
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Optional: Print loss every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f'PyTorch Epoch {epoch+1}/{max_iter}, Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test_tensor).squeeze()
    test_preds = (test_outputs > 0.5).numpy()
    pytorch_acc = np.mean(test_preds == y_test.flatten()) * 100
print(f"PyTorch MLP Test Accuracy: {pytorch_acc:.2f}%")

# Sonuçları Karşılaştırma
print(f"En iyi model: {max([(acc_2layer, '2-Layer MLP'), (acc_3layer, '3-Layer MLP'), (sklearn_acc, 'Scikit-learn MLP'), (pytorch_acc, 'PyTorch MLP')], key=lambda x: x[0])[1]}")