"""
Simplest Machine Learning Example: 1D Linear Regression from scratch (no dependencies)

Scenario: Predict exam score (label) from study hours (feature) using a linear model.
Model: y_hat = weight * x + bias

This script trains the model using gradient descent to minimize Mean Squared Error (MSE).
"""

from typing import List, Tuple


def predict(feature_value: float, weight: float, bias: float) -> float:
    """Return the model output y_hat for a single feature value.

    The model is y_hat = weight * x + bias.
    """
    return weight * feature_value + bias


def compute_predictions(features: List[float], weight: float, bias: float) -> List[float]:
    """Vectorized prediction for a list of features."""
    return [predict(x, weight, bias) for x in features]


def compute_mean_squared_error(labels: List[float], predictions: List[float]) -> float:
    """Compute the mean squared error between labels and predictions."""
    assert len(labels) == len(predictions), "labels and predictions must have the same length"
    num_examples: int = len(labels)
    if num_examples == 0:
        return 0.0
    squared_errors_sum: float = 0.0
    for y_true, y_pred in zip(labels, predictions):
        difference: float = y_pred - y_true
        squared_errors_sum += difference * difference
    return squared_errors_sum / float(num_examples)


def compute_gradients(
    features: List[float], labels: List[float], weight: float, bias: float
) -> Tuple[float, float]:
    """Compute gradients of MSE w.r.t. weight and bias.

    For MSE = (1/n) * sum((y_hat - y)^2), where y_hat = w*x + b, the gradients are:
      dMSE/dw = (2/n) * sum((y_hat - y) * x)
      dMSE/db = (2/n) * sum(y_hat - y)
    """
    assert len(features) == len(labels), "features and labels must have the same length"
    num_examples: int = len(features)
    if num_examples == 0:
        return 0.0, 0.0

    gradient_weight_sum: float = 0.0
    gradient_bias_sum: float = 0.0
    for x_value, y_true in zip(features, labels):
        y_pred: float = predict(x_value, weight, bias)
        error: float = y_pred - y_true
        gradient_weight_sum += error * x_value
        gradient_bias_sum += error

    factor: float = 2.0 / float(num_examples)
    d_weight: float = factor * gradient_weight_sum
    d_bias: float = factor * gradient_bias_sum
    return d_weight, d_bias


def train_linear_regression(
    features: List[float],
    labels: List[float],
    learning_rate: float = 0.01,
    epochs: int = 1000,
    initial_weight: float = 0.0,
    initial_bias: float = 0.0,
) -> Tuple[float, float, List[float]]:
    """Train a 1D linear regression model with gradient descent.

    Returns the learned weight, bias, and the loss history for inspection.
    """
    weight: float = initial_weight
    bias: float = initial_bias
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        predictions: List[float] = compute_predictions(features, weight, bias)
        loss: float = compute_mean_squared_error(labels, predictions)
        loss_history.append(loss)

        d_weight, d_bias = compute_gradients(features, labels, weight, bias)

        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        if epoch % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | loss={loss:.6f} | weight={weight:.6f} | bias={bias:.6f}"
            )

    return weight, bias, loss_history


def main() -> None:
    # Training data: study hours (feature) -> exam score (label)
    # We intentionally choose a perfectly linear relationship: y = 5*x + 10
    features: List[float] = [1, 2, 3, 4, 5]
    labels: List[float] = [15, 20, 25, 30, 35]

    print("Training a linear model y_hat = weight * x + bias")
    print(f"Features (x): {features}")
    print(f"Labels   (y): {labels}")

    learned_weight, learned_bias, _ = train_linear_regression(
        features=features,
        labels=labels,
        learning_rate=0.01,
        epochs=1000,
        initial_weight=0.0,
        initial_bias=0.0,
    )

    print("\nTraining complete.")
    print(f"Learned weight: {learned_weight:.6f}")
    print(f"Learned bias:   {learned_bias:.6f}")

    # Demonstrate prediction (model output) for a new input
    new_study_hours: float = 6.0
    predicted_score: float = predict(new_study_hours, learned_weight, learned_bias)
    print(f"\nFor x = {new_study_hours:.1f} study hours, predicted exam score (output) = {predicted_score:.2f}")


if __name__ == "__main__":
    main()

