import numpy as np

# Function to perform gradient descent
def gradient_descent(x, y):
    # Initialize the parameters
    m_curr = 0
    b_curr = 0
    iterations = 10000
    n = len(x)  # Number of data points
    learning_rate = 0.0001

    # Gradient descent loop
    for i in range(iterations):
        # Calculate the predicted y values
        y_predicted = m_curr * x + b_curr
        
        # Calculate the cost (mean squared error)
        cost = (1/n) * sum((y - y_predicted) ** 2)
        
        # Calculate the gradients
        md = -(2/n) * sum(x * (y - y_predicted))  # Gradient with respect to m
        bd = -(2/n) * sum(y - y_predicted)  # Gradient with respect to b
        
        # Update the parameters using the gradients
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        # Print the values of m, b, cost, and iteration number
        print("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))

# Sample data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Run the gradient descent algorithm
gradient_descent(x, y)
