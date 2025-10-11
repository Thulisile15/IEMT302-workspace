"""
A minimal from-scratch linear regression example using gradient descent.

This script generates synthetic data from a true linear relationship
    y = true_weight * x + true_bias + noise
and then learns parameters (weight and bias) by minimizing mean squared error
via gradient descent.

Only NumPy is used for numerical computations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    num_steps: int = 2000
    print_every: int = 200
    seed: int = 42


@dataclass
class Dataset:
    x: np.ndarray  # shape: (num_points,)
    y: np.ndarray  # shape: (num_points,)


@dataclass
class ModelParams:
    weight: float
    bias: float


def generate_synthetic_data(
    num_points: int = 100,
    true_weight: float = 3.0,
    true_bias: float = 2.0,
    noise_std: float = 1.0,
    seed: int | None = 42,
) -> Dataset:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x = rng.uniform(low=-3.0, high=3.0, size=num_points)
    noise = rng.normal(loc=0.0, scale=noise_std, size=num_points)
    y = true_weight * x + true_bias + noise
    return Dataset(x=x.astype(np.float64), y=y.astype(np.float64))


def compute_predictions(x: np.ndarray, params: ModelParams) -> np.ndarray:
    return params.weight * x + params.bias


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def compute_gradients(x: np.ndarray, y: np.ndarray, params: ModelParams) -> Tuple[float, float]:
    predictions = compute_predictions(x, params)
    errors = predictions - y
    n = x.shape[0]
    grad_weight = (2.0 / n) * float(np.dot(errors, x))
    grad_bias = (2.0 / n) * float(np.sum(errors))
    return grad_weight, grad_bias


def train_linear_regression(
    data: Dataset,
    config: TrainingConfig = TrainingConfig(),
    init_params: ModelParams | None = None,
) -> Tuple[ModelParams, List[float]]:
    if init_params is None:
        # Small random initialization for stability
        rng = np.random.default_rng(config.seed)
        params = ModelParams(weight=float(rng.normal(0.0, 0.1)), bias=float(rng.normal(0.0, 0.1)))
    else:
        params = ModelParams(weight=init_params.weight, bias=init_params.bias)

    history: List[float] = []

    for step in range(config.num_steps + 1):
        predictions = compute_predictions(data.x, params)
        loss = mean_squared_error(data.y, predictions)
        history.append(loss)

        should_print = (step % config.print_every == 0) or (step == config.num_steps)
        if should_print:
            print(
                f"step={step:4d}  loss={loss:10.6f}  weight={params.weight:+8.4f}  bias={params.bias:+8.4f}"
            )

        if step == config.num_steps:
            break

        grad_w, grad_b = compute_gradients(data.x, data.y, params)
        params.weight -= config.learning_rate * grad_w
        params.bias -= config.learning_rate * grad_b

    return params, history


def main() -> None:
    # Ground-truth linear relationship used to generate data
    true_weight = 3.0
    true_bias = 2.0

    print("Generating synthetic data...")
    data = generate_synthetic_data(
        num_points=100,
        true_weight=true_weight,
        true_bias=true_bias,
        noise_std=1.0,
        seed=42,
    )

    config = TrainingConfig(learning_rate=0.05, num_steps=2000, print_every=200, seed=42)

    print(
        f"True parameters: weight={true_weight:+.4f}, bias={true_bias:+.4f}\n"
        f"Training with learning_rate={config.learning_rate}, steps={config.num_steps}\n"
    )

    learned_params, history = train_linear_regression(data, config=config)

    print("\nTraining complete.")
    print(f"Learned parameters: weight={learned_params.weight:+.4f}, bias={learned_params.bias:+.4f}")

    # Evaluate on a few inputs
    test_x = np.array([-2.0, 0.0, 1.5, 4.0], dtype=np.float64)
    preds = compute_predictions(test_x, learned_params)
    truth = true_weight * test_x + true_bias

    print("\nSample predictions (x -> predicted | true):")
    for x_val, y_hat, y_true in zip(test_x, preds, truth):
        print(f"  {x_val:+5.2f} -> {y_hat:+8.4f} | {y_true:+8.4f}")


if __name__ == "__main__":
    main()
