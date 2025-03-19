import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef, cohen_kappa_score
)

def evaluate_model(y_true, y_pred, y_prob=None, dataset_type=None):
    """
    Modelin temel değerlendirme metriklerini hesaplar.
    """
    if dataset_type:
        print(f"Evaluation for {dataset_type} Dataset:")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    
    # ROC-AUC skoru ve Log Loss için olasılıklar gerekiyorsa hesapla
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    logloss = log_loss(y_true, y_prob) if y_prob is not None else None
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    if logloss is not None:
        print(f"Log Loss: {logloss:.4f}")
    
    print("\nClassification Report:")
    print(cr)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "kappa": kappa,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "confusion_matrix": cm,
        "classification_report": cr
    }

def plot_confusion_matrix(y_true, y_pred, labels=["Class 0", "Class 1"]):
    """
    Confusion matrix'i çizdirir.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def plot_learning_curve(train_costs, test_costs):
    """
    Modelin öğrenme eğrisini (cost vs. iteration) çizdirir.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_costs, label="Training Loss", color="blue")
    plt.plot(test_costs, label="Test Loss", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

def save_results(y_true, y_pred, filename="results.csv"):
    """
    Model tahminlerini ve gerçek etiketleri CSV olarak kaydeder.
    """
    df = pd.DataFrame({"True Label": y_true, "Predicted Label": y_pred})
    df.to_csv(filename, index=False)
    print(f"Sonuçlar {filename} dosyasına kaydedildi!")


