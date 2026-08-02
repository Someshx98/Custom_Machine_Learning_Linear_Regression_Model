import numpy as np


def standardize(x):
    """Rescales x to have mean 0 and standard deviation 1.

    Gradient descent is very sensitive to the scale of the input data:
    with raw, unscaled features, a learning rate that works nicely on one
    dataset can explode into NaNs on another (or crawl painfully slowly).
    Standardizing first means one learning-rate range works well no
    matter what units your data is in.
    """
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        # Every value is identical - nothing to scale, avoid a divide-by-zero.
        return np.zeros_like(x, dtype=float), mean, 1.0
    return (x - mean) / std, mean, std


def predict(x, m, b):
    """Takes an input(x), applies our current slope(m) and intercept(b),
    and returns the engine's prediction."""
    return (m * np.asarray(x, dtype=float)) + b


def calculate_cost(x_data, y_data, m, b):
    """Compares predictions against the actual y_data and returns the
    Mean Squared Error (MSE). Vectorized with NumPy so it stays fast even
    on large datasets, instead of looping row-by-row in pure Python."""
    errors = y_data - predict(x_data, m, b)
    return np.mean(errors ** 2)


def update_weights(x_data, y_data, m, b, learning_rate):
    """Calculates gradients and updates m and b to move down the cost
    valley (one step of batch gradient descent)."""
    errors = y_data - predict(x_data, m, b)
    m_deriv = -2 * np.mean(x_data * errors)
    b_deriv = -2 * np.mean(errors)

    m -= m_deriv * learning_rate
    b -= b_deriv * learning_rate

    return m, b


def train(x_data, y_data, m, b, learning_rate, iterations):
    """Runs the gradient descent learning loop for a set number of
    iterations and returns the learned (m, b) plus the cost history.

    Internally standardizes x_data before training for numerical
    stability, then converts the learned weights back to the original
    scale, so callers can keep working in their data's real units. If an
    update ever blows up (learning rate too high), training stops early
    and keeps the last stable weights instead of returning NaN/inf.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_scaled, x_mean, x_std = standardize(x_data)

    cost_history = []
    for _ in range(iterations):
        new_m, new_b = update_weights(x_scaled, y_data, m, b, learning_rate)
        cost = calculate_cost(x_scaled, y_data, new_m, new_b)
        if not np.isfinite(cost):
            # This learning rate is too high for stable convergence; keep
            # the last good weights instead of exploding to NaN/inf.
            break
        m, b = new_m, new_b
        cost_history.append(cost)

    # Undo the standardization: y = m*((x-mean)/std) + b becomes
    # y = (m/std)*x + (b - m*mean/std) in the original x units.
    m_original = m / x_std
    b_original = b - (m * x_mean / x_std)

    return m_original, b_original, cost_history


def train_test_split(x_data, y_data, test_size=0.2, random_state=42):
    """Randomly splits data into a training set and a held-out test set,
    so the model can be graded on data it never saw while learning."""
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    n = len(x_data)

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n)

    split_point = int(round(n * (1 - test_size)))
    split_point = min(max(split_point, 1), n - 1)  # keep both sides non-empty

    train_idx, test_idx = shuffled[:split_point], shuffled[split_point:]
    return x_data[train_idx], x_data[test_idx], y_data[train_idx], y_data[test_idx]


def calculate_r2(y_data, prediction):
    """Calculates the R-squared score to grade the model's accuracy."""
    y_data = np.asarray(y_data, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    y_mean = np.mean(y_data)
    ss_tot = np.sum((y_data - y_mean) ** 2)  # Total Sum of Squares
    ss_res = np.sum((y_data - prediction) ** 2)  # Residual Sum of Squares

    if ss_tot == 0:
        # Every actual y value is identical - R^2 is undefined in the
        # usual sense, so just report whether the fit is perfect.
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)