## Simplest Machine Learning: 1D Linear Regression from Scratch

This repo contains a single, minimal example of machine learning implemented from scratch in pure Python. It demonstrates all the core ingredients you were asked to show: training data, features, labels, a model with weight and bias, training via gradient descent, and producing outputs (predictions).

### Scenario
Predict a student's exam score from the number of study hours.

- **feature (x)**: hours studied
- **label (y)**: exam score
- **model**: y_hat = weight * x + bias

We will learn the best `weight` (slope) and `bias` (intercept) that map hours studied to exam score.

### Training data
We use a small toy dataset with a perfectly linear relationship y = 5x + 10:

| hours (x) | score (y) |
|-----------|-----------|
| 1         | 15        |
| 2         | 20        |
| 3         | 25        |
| 4         | 30        |
| 5         | 35        |

### The model
We assume a linear relationship between input and output:

- **Prediction (output)**: y_hat = weight * x + bias
- **weight (w)**: how much the prediction changes per extra hour studied
- **bias (b)**: baseline score when x = 0 (the intercept)

### Loss function (what we minimize)
We train the model by minimizing Mean Squared Error (MSE) between predictions and labels:

MSE = (1/n) * Σ (y_hat - y)^2

This is small when predictions y_hat are close to the true labels y.

### How training works (gradient descent)
We start with initial values for `weight` and `bias` (e.g., both zero). On each training step, we:

1. Compute predictions y_hat for all training examples using the current `weight` and `bias`.
2. Compute the loss (MSE) between predictions and true labels.
3. Compute the gradients (how to change parameters to reduce loss):
   - dMSE/dw = (2/n) * Σ ((y_hat - y) * x)
   - dMSE/db = (2/n) * Σ (y_hat - y)
4. Update parameters in the direction that reduces loss:
   - weight ← weight − learning_rate * dMSE/dw
   - bias   ← bias   − learning_rate * dMSE/db

Repeat steps 1–4 for many epochs until the loss is small and the parameters settle near their optimal values.

### What each piece means in this example
- **Training data**: the table above of (hours, score) pairs
- **Features**: the list of hours `[1, 2, 3, 4, 5]`
- **Labels**: the list of scores `[15, 20, 25, 30, 35]`
- **Weight**: the learned slope, expected near 5.0
- **Bias**: the learned intercept, expected near 10.0
- **Output**: a prediction y_hat for any given x using y_hat = weight*x + bias

---

## How to run it

### Prerequisites
- Python 3.8+ (no external libraries needed)

### Steps
1. Open a terminal in the repo root.
2. (Optional but recommended) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Run the example:

```bash
python3 main.py
```

You should see training progress logs and then the learned parameters and a sample prediction.

### Example output (your numbers may differ slightly)

```text
Training a linear model y_hat = weight * x + bias
Features (x): [1, 2, 3, 4, 5]
Labels   (y): [15, 20, 25, 30, 35]
Epoch  100/1000 | loss=0.036xxx | weight=4.3xxx | bias=10.3xxx
...
Epoch 1000/1000 | loss=0.000000 | weight=5.000000 | bias=10.000000

Training complete.
Learned weight: 5.000000
Learned bias:   10.000000

For x = 6.0 study hours, predicted exam score (output) = 40.00
```

If you prefer to change how long training runs or how fast it learns, open `main.py` and adjust `epochs` and `learning_rate` in the `train_linear_regression(...)` call.

---

## Why this is machine learning
We did not program the exact formula in advance; instead, we gave the model examples (training data) and let it learn the best `weight` and `bias` by minimizing error. After training, the model generalizes to new inputs (e.g., 6 study hours) and produces an output (predicted exam score).

# IEMT302-workspace