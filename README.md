## Simple From-Scratch Linear Regression (Minimal ML Example)

This repository demonstrates the simplest possible machine learning example: fitting a straight line to data using gradient descent, implemented from scratch with only NumPy.

- **Goal**: Recover the parameters of a line that generated synthetic data: `y = w * x + b + noise`.
- **Learning**: We adjust `w` (weight) and `b` (bias) to minimize Mean Squared Error (MSE) between predictions and observed `y`.
- **Why this is ML**: The model learns parameters from data by optimizing a loss function.

### What the code does
The script `simple_ml.py`:
1. Generates synthetic data from a true line (`w=3.0`, `b=2.0`) with Gaussian noise.
2. Initializes the model parameters (`w`, `b`) randomly.
3. Repeatedly computes predictions, calculates MSE loss, computes gradients analytically, and applies gradient descent updates.
4. Prints training progress and shows sample predictions compared to the true line.

### Mathematical details (concise)
- **Model**: \( \hat{y} = w x + b \)
- **Loss**: Mean Squared Error, \( L = \frac{1}{n} \sum_i (\hat{y}_i - y_i)^2 \)
- **Gradients**:
  - \( \frac{\partial L}{\partial w} = \frac{2}{n} \sum_i (\hat{y}_i - y_i) x_i \)
  - \( \frac{\partial L}{\partial b} = \frac{2}{n} \sum_i (\hat{y}_i - y_i) \)
- **Update rule (gradient descent)**: \( w \leftarrow w - \eta \frac{\partial L}{\partial w},\quad b \leftarrow b - \eta \frac{\partial L}{\partial b} \) where \( \eta \) is the learning rate.

### Requirements
- Python 3.8+
- NumPy
- Matplotlib (only if you want plots)

Install with:
```bash
python3 -m pip install -r requirements.txt
```

### How to run
From the repo root:
```bash
python3 simple_ml.py
```

You should see output like:
```text
Generating synthetic data...
True parameters: weight=+3.0000, bias=+2.0000
Training with learning_rate=0.05, steps=2000

step=   0  loss= 28.125819  weight= +0.0305  bias= -0.1040
step= 200  loss=  0.958864  weight= +3.0128  bias= +1.9870
...
step=2000  loss=  0.958864  weight= +3.0128  bias= +1.9870

Training complete.
Learned parameters: weight=+3.0128, bias=+1.9870

Sample predictions (x -> predicted | true):
  -2.00 ->  -4.0385 |  -4.0000
  +0.00 ->  +1.9870 |  +2.0000
  +1.50 ->  +6.5062 |  +6.5000
  +4.00 -> +14.0381 | +14.0000
```

### Plotting (optional)
Install plotting dependencies if you haven't already:
```bash
python3 -m pip install -r requirements.txt
```

Produce plots without opening windows (saved to `plots/`):
```bash
python3 simple_ml.py --plot --save plots
```

Show interactive windows (if your environment supports a GUI):
```bash
python3 simple_ml.py --plot --show
```

Two PNGs are produced when saving:
- `plots/loss.png`: training loss over steps
- `plots/fit.png`: data points, learned line, and true line

### Explanation in plain words
- We start with a guess for `w` and `b`. The model predicts a value for each input `x`.
- We measure how wrong the predictions are using MSE. Larger errors increase the loss.
- The gradients tell us which way to change `w` and `b` to reduce the loss the fastest.
- Repeating small steps in that direction (gradient descent) nudges `w` and `b` toward values that best fit the data.
- Because the data was created from a true line, the learned parameters end up very close to the true `w=3.0`, `b=2.0`.

### Project structure
- `simple_ml.py`: Minimal implementation of data generation and gradient descent training
- `requirements.txt`: Python dependency list
- `.gitignore`: Standard Python ignores

### Repro tips
- If you need reproducible runs, parameters and data are seeded in the script; you can change the `seed` in `TrainingConfig` or in the data generator for different randomness.
- If training seems unstable, lower the learning rate in `TrainingConfig`.

### Next steps (optional)
- Plot training loss over time using matplotlib.
- Switch to closed-form solution (normal equation) to compare with gradient descent.
- Extend to multivariate linear regression with vectorized parameters.
