import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from models.mlp_3layer import MLPClassifier  # 3 katmanlı model import edildi
from experiments.results_analysis import evaluate_model, plot_confusion_matrix, plot_learning_curve, save_results

# Veri yükleme ve ayırma
df = pd.read_csv("MLP/data/BankNote_Authentication.csv").sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy().reshape(-1, 1)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3 katmanlı MLP modeli oluşturma ve eğitim
mlp = MLPClassifier(n_x=X_train.shape[1], n_h1=10, n_h2=6)  # Ekstra katman eklendi
mlp.fit(X_train, y_train, X_test, y_test, n_steps=5000)

# Tahmin yapma
y_pred_train = mlp.predict(X_train).flatten()
y_pred_test = mlp.predict(X_test).flatten()
y_train_true = y_train.flatten()
y_test_true = y_test.flatten()

# Modeli değerlendirme
evaluate_model(y_train_true, y_pred_train, dataset_type="Training")
evaluate_model(y_test_true, y_pred_test, dataset_type="Test")

# Confusion matrix çizdirme
plot_confusion_matrix(y_test_true, y_pred_test)
plot_confusion_matrix(y_train_true, y_pred_train)

# Öğrenme eğrisi çizdirme
plot_learning_curve(mlp.train_costs, mlp.test_costs)

# Sonuçları kaydetme
save_results(y_test_true, y_pred_test, "MLP/experiments/test_results.csv")
