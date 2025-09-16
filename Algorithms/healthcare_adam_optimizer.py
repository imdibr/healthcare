
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

class AdamOptimizer:
    """
    Adam Optimizer for Healthcare Disease Prediction
    Adaptive Moment Estimation with medical-specific modifications
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss function"""
        # Add small epsilon to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_medical_metrics(self, y_true, y_pred):
        """Compute medical-specific metrics"""
        y_pred_binary = (y_pred >= 0.5).astype(int)

        TP = np.sum((y_true == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true == 1) & (y_pred_binary == 0))

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision
        }

    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the model using Adam optimization
        """
        m, n = X.shape

        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0

        # Initialize Adam parameters
        m_w = np.zeros(n)
        v_w = np.zeros(n)
        m_b = 0
        v_b = 0

        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'sensitivity': [],
            'specificity': []
        }

        for epoch in range(epochs):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute loss
            loss = self.compute_loss(y, predictions)

            # Compute gradients
            dw = (1/m) * X.T.dot(predictions - y)
            db = (1/m) * np.sum(predictions - y)

            # Update biased first and second moments
            m_w = self.beta1 * m_w + (1 - self.beta1) * dw
            v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
            m_b = self.beta1 * m_b + (1 - self.beta1) * db
            v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)

            # Bias correction
            m_w_corrected = m_w / (1 - self.beta1 ** (epoch + 1))
            v_w_corrected = v_w / (1 - self.beta2 ** (epoch + 1))
            m_b_corrected = m_b / (1 - self.beta1 ** (epoch + 1))
            v_b_corrected = v_b / (1 - self.beta2 ** (epoch + 1))

            # Update parameters
            self.weights -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            self.bias -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)

            # Compute metrics
            metrics = self.compute_medical_metrics(y, predictions)

            # Store history
            self.history['loss'].append(loss)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['sensitivity'].append(metrics['sensitivity'])
            self.history['specificity'].append(metrics['specificity'])

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {metrics['accuracy']:.4f} | "
                      f"Sens: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f}")

    def predict(self, X):
        """Make predictions"""
        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)

    def predict_binary(self, X):
        """Make binary predictions"""
        return (self.predict(X) >= 0.5).astype(int)

    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss
        axes[0, 0].plot(self.history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(self.history['accuracy'])
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)

        # Sensitivity
        axes[1, 0].plot(self.history['sensitivity'])
        axes[1, 0].set_title('Sensitivity (Recall)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Sensitivity')
        axes[1, 0].grid(True)

        # Specificity
        axes[1, 1].plot(self.history['specificity'])
        axes[1, 1].set_title('Specificity')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Specificity')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run heart disease prediction
    """
    print("Heart Disease Prediction using Adam Optimizer")
    print("=" * 50)

    # Load the dataset
    try:
        df = pd.read_csv('heart_disease_data.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: heart_disease_data.csv not found!")
        return

    # Prepare the data
    X = df.drop('heart_disease', axis=1).values
    y = df['heart_disease'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the features (important for medical data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")

    # Create and train the model
    model = AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)

    print("\nTraining the model...")
    model.fit(X_train_scaled, y_train, epochs=1000, verbose=True)

    # Make predictions
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    train_binary = model.predict_binary(X_train_scaled)
    test_binary = model.predict_binary(X_test_scaled)

    # Evaluate the model
    train_metrics = model.compute_medical_metrics(y_train, train_predictions)
    test_metrics = model.compute_medical_metrics(y_test, test_predictions)

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    # Plot training history
    model.plot_training_history()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\nConfusion Matrix Analysis:")
    TP, FN, FP, TN = cm.ravel()
    print(f"True Positives (TP): {TP}")
    print(f"False Negatives (FN): {FN}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")

    return model, scaler

if __name__ == "__main__":
    model, scaler = main()
