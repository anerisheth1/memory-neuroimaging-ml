# smote_logistic_regression.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from imblearn.over_sampling import SMOTE


from data_loader import load_data
from pca_model_visualizer import pca_analysis, visualize_pca


def smote_logistic_regression_analysis(X_pca, y):
    """
    Splits data, applies SMOTE oversampling to the training set,
    trains Logistic Regression, and evaluates performance.
    """

    # 1. Train/Test Split (Stratified to maintain class ratio in split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Class counts check
    print("\n--- SMOTE Class Counts ---")
    print(f"Original Training Class Counts (0:Non-Mem, 1:Mem): {np.bincount(y_train)}")
    print(f"SMOTE Training Class Counts (0:Non-Mem, 1:Mem):    {np.bincount(y_train_smote)}")

    # 3. Train Logistic Regression
    clf = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        C=50.0  # Use a moderately higher C for less regularization
    )
   
    clf.fit(X_train_smote, y_train_smote)

    # 4. Predict and Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Use the default prediction threshold (0.5)
    y_pred = clf.predict(X_test) 

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nSMOTE Logistic Regression Performance:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["Non-Memory", "Memory"],
                yticklabels=["Non-Memory", "Memory"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - SMOTE Logistic Regression")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - SMOTE Logistic Regression")
    plt.legend()
    plt.show()

    return clf, (acc, prec, rec, f1, auc_score)


if __name__ == "__main__":
    # run PCA
    pca_model, X_reduced = pca_analysis(n_components=20)

    # Visualize PCA map 
    visualize_pca(pca_model)

    X, y = load_data()

    clf, metrics = smote_logistic_regression_analysis(X_reduced, y)