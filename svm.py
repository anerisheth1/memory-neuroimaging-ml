# svm_classifier.py
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

from data_loader import load_data
from pca_model_visualizer import pca_analysis, visualize_pca


def svm_analysis(X_pca, y):
    """Train/test split, linear SVM classifier + evaluation metrics and plots."""

    # --- 1. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- 2. Train Linear SVM ---
    svm_clf = SVC(
        kernel='linear',  # enables interpretability (weights) 
        probability=True,  # needed for ROC curves
    )
    svm_clf.fit(X_train, y_train)

    # --- 3. Predictions ---
    y_prob = svm_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.4).astype(int)  # TRY THRESHOLDS 0.4, 0.35, 0.3


    # --- 4. Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nSVM Performance:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # --- 5. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=["Non-Memory", "Memory"],
                yticklabels=["Non-Memory", "Memory"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - SVM")
    plt.show()

    # --- 6. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2, color="purple")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - SVM")
    plt.legend()
    plt.show()

    return svm_clf, (acc, prec, rec, f1, auc_score)


if __name__ == "__main__":
    # run PCA
    pca_model, X_reduced = pca_analysis(n_components=5)

    # Visualize PCA spatial map 
    visualize_pca(pca_model)

    # Load y labels 
    X, y = load_data()

    # Run SVM
    svm_clf, metrics = svm_analysis(X_reduced, y)
