# logistic_regression.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from saliency_map import saliency_brain_map


from data_loader import load_data
from pca_model_visualizer import pca_analysis, visualize_pca


def logistic_regression_analysis(X_pca, y):
    """Train/test split, logistic regression, evaluation metrics + plots."""

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    new_threshold = 0.1

    y_pred = (y_prob >= new_threshold).astype(int)


    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nLogistic Regression Performance:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Memory", "Memory"],
                yticklabels=["Non-Memory", "Memory"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.show()

    return clf, (acc, prec, rec, f1, auc_score)


if __name__ == "__main__":
    # --- Run PCA first ---
    pca_model, X_reduced = pca_analysis(n_components=10)

    # --- Visualize PCA map ---
    visualize_pca(pca_model)

    # --- Load labels to match X_reduced ---
    X, y = load_data()

    # --- Run Logistic Regression ---
    clf, metrics = logistic_regression_analysis(X_reduced, y)

    # For Logistic Regression
    saliency_brain_map(clf, pca_model, title="Logistic Regression Saliency Map")
