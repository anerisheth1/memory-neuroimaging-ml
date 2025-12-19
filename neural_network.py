import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight


from data_loader import load_data
from saliency_map import nn_saliency_map



def neural_network(X_full, y):
    """
    Trains and evaluates a simple Multi-Layer Perceptron (MLP) 
    with Dropout regularization and class weights for improved stability.
    """

    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 2. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Calculate Class Weights (Though balanced, this ensures the loss function is priority-aware)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    print(f"\nCalculated Class Weights: {class_weights}")


    # 4. Build the IMPROVED MLP Model
    input_dim = X_train_scaled.shape[1]
    
    model = Sequential([
        # Hidden Layer 1 (Reduced Complexity)
        Dense(32, activation='relu', input_shape=(input_dim,)),
        # Dropout Layer (Regularization)
        Dropout(0.4), # Drop 40% of neurons randomly during training
        # Hidden Layer 2 (Reduced Complexity)
        Dense(16, activation='relu'),
        # Output Layer
        Dense(1, activation='sigmoid')
    ])

    # 5. Compile and Train the Model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\n--- Training Improved Neural Network ---")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,         # Increased epochs due to Dropout
        batch_size=8,
        verbose=0,
        validation_split=0.1,
        class_weight=class_weights # Apply class weights
    )

    # 6. Predict and Evaluate
    y_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_prob > 0.5).astype(int) 

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nImproved Neural Network Performance:")
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
    plt.title("Confusion Matrix - Improved Neural Network")
    plt.show()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Improved Neural Network")
    plt.legend()
    plt.show()

    return model, scaler, (acc, prec, rec, f1, auc_score)


if __name__ == "__main__":
    # Load the full, high-dimensional data
    X, y = load_data() 
    
    # Run Improved Neural Network analysis on full data
    clf, scaler, metrics = neural_network(X, y)
    idx = np.where(y == 1)[0][0]  # first memory example
    
    sample_scaled = scaler.transform(X[idx].reshape(1, -1))

    saliency = nn_saliency_map(clf, sample_scaled, title="Neural Network Saliency Map (Memory)")