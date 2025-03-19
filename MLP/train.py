import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from models.mlp_2layer import MLPClassifier
from experiments.results_analysis import evaluate_model, plot_confusion_matrix, plot_learning_curve, save_results

# Veri yükleme ve ayırma
df = pd.read_csv("MLP/data/BankNote_Authentication.csv").sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy().reshape(-1, 1)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model oluşturma ve eğitim (ReLU aktivasyonu ile)
mlp_relu = MLPClassifier(n_x=X_train.shape[1], n_h=6, activation_function = 'relu')
mlp_relu.fit(X_train, y_train, X_test, y_test, n_steps=5000)

# Model oluşturma ve eğitim (tanh aktivasyonu ile)
mlp_tanh = MLPClassifier(n_x=X_train.shape[1], n_h=6, activation_function ='tanh')
mlp_tanh.fit(X_train, y_train, X_test, y_test, n_steps=5000)

# Tahmin yapma (ReLU)
y_pred_train_relu = mlp_relu.predict(X_train).flatten()
y_pred_test_relu = mlp_relu.predict(X_test).flatten()

# Tahmin yapma (tanh)
y_pred_train_tanh = mlp_tanh.predict(X_train).flatten()
y_pred_test_tanh = mlp_tanh.predict(X_test).flatten()

# Gerçek değerler
y_train_true = y_train.flatten()
y_test_true = y_test.flatten()

# Model değerlendirme (ReLU)
evaluate_model(y_train_true, y_pred_train_relu, dataset_type="Training (ReLU)")
evaluate_model(y_test_true, y_pred_test_relu, dataset_type="Test (ReLU)")

# Model değerlendirme (tanh)
evaluate_model(y_train_true, y_pred_train_tanh, dataset_type="Training (tanh)")
evaluate_model(y_test_true, y_pred_test_tanh, dataset_type="Test (tanh)")

# Confusion matrix çizdirme
plot_confusion_matrix(y_test_true, y_pred_test_relu)
plot_confusion_matrix(y_train_true, y_pred_train_relu)
plot_confusion_matrix(y_test_true, y_pred_test_tanh)
plot_confusion_matrix(y_train_true, y_pred_train_tanh)

# Öğrenme eğrisi çizdirme
plot_learning_curve(mlp_relu.train_costs, mlp_relu.test_costs)
plot_learning_curve(mlp_tanh.train_costs, mlp_tanh.test_costs)

# Sonuçları kaydetme
save_results(y_test_true, y_pred_test_relu, "MLP/experiments/test_results_relu.csv")  
save_results(y_test_true, y_pred_test_tanh, "MLP/experiments/test_results_tanh.csv")