import numpy as np

def predict(x, m, b):
    """Takes an input(x), applies our current slope(m), and intercept(b) and returns the engine's prediction."""
    return (m * x) + b

def calculate_cost(x_data, y_data, m, b):
    """Loops through all data points, compares the predictions against the actual y_data, and calculates the Mean Squared Error (MSE)."""
    sum = 0.0
    for i in range(len(x_data)):
        sum += (y_data[i] - ((m * x_data[i]) + b)) ** 2
    return sum / len(y_data)

def update_weights(x_data, y_data, m, b, learning_rate):
    """Calculates gradients and updates m and b to move down the cost valley."""
    m_deriv = 0.0
    b_deriv = 0.0
    N = len(x_data)
    for i in range(N):
        prediction = predict(x_data[i], m, b)
        m_deriv += -2 * x_data[i] * (y_data[i] - prediction)
        b_deriv += -2 * (y_data[i] - prediction)
        
    m -= (m_deriv / N) * learning_rate
    b -= (b_deriv / N) * learning_rate
    
    return m, b

def train(x_data, y_data, m, b, learning_rate, iterations):
    """Runs the learning loop for a set number of iterations."""
    cost_history = []
    for i in range(iterations):
        m, b = update_weights(x_data, y_data, m, b, learning_rate)
        cost = calculate_cost(x_data, y_data, m, b)
        cost_history.append(cost)
        
    return m, b, cost_history

def calculate_r2(y_data, prediction):
    """Calculates the R-squared score to grade the model's accuracy."""
    y_mean = np.mean(y_data)
    ss_tot = 0.0 # Total Sum of Squares
    ss_res = 0.0 # Residual Sum of Squares
    for i in range(len(y_data)):
        ss_tot += (y_data[i] - y_mean) ** 2
        ss_res += (y_data[i] - prediction[i]) ** 2
        
    res = 1 - (ss_res / ss_tot)
    return res