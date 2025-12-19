# l1_logistic_regression_full_data.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from saliency_map import saliency_full_voxel_map

from data_loader import load_data


def l1_logistic_regression_full_data(X_full, y):
    """
    Uses the full feature set (no PCA) and L1 regularization for feature selection.
    
    Args:
        X_full (np.array): The high-dimensional original feature matrix (no PCA).
        y (np.array): The labels (0 or 1).
    """

    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.25, random_state=42, stratify=y
    )
    
    X_train_final, y_train_final = X_train, y_train

    # 2. Train L1 Logistic Regression (Lasso)
    print("\n--- Training L1 Logistic Regression on Full Feature Set ---")

    clf = LogisticRegression(
        # penalty to 'l1' for Lasso feature selection
        penalty='l1',
        solver='liblinear',
        max_iter=10000, # Increased max_iter for convergence stability
        # C - inverse of regularization strength. 
        # C=1.0 is default,  
        # C=10.0 or C=50.0 to make the model less penalized (more complex)
        C=50.0 
    )
    
    clf.fit(X_train_final, y_train_final)

    # 3. Predict and Evaluate (using the original test data)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test) 

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nL1 Logistic Regression Performance (Full Data):")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=["Non-Memory", "Memory"],
                yticklabels=["Non-Memory", "Memory"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - L1 Logistic Regression (Full Data)")
    plt.show()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - L1 Logistic Regression (Full Data)")
    plt.legend()
    plt.show()

    return clf, (acc, prec, rec, f1, auc_score)


if __name__ == "__main__":
    # full, high-dimensional data
    X, y = load_data() 
    
    # L1 Logistic Regression on full data
    clf, metrics = l1_logistic_regression_full_data(X, y)

    saliency_full_voxel_map(clf, title="L1 Logistic Regression Saliency Map (Full Data)")

    # Analyze which original features were kept (non-zero weights)
    n_selected = np.sum(clf.coef_ != 0)
    print(f"Number of features selected by L1: {n_selected}")