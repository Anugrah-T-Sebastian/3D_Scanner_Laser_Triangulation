import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
Y = np.array([1.48, 3.68, 8.94, 9.36, 14])
X = np.array([10.624730492162254, 24.845142675791827, 55.76918980683254, 57.79133423763752, 90.29026037917573])

# Define the functions to fit
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit the curves
popt_linear, _ = curve_fit(linear_func, X, Y)
popt_quadratic, _ = curve_fit(quadratic_func, X, Y)
popt_cubic, _ = curve_fit(cubic_func, X, Y)

# Generate curve values for plotting
x_curve = np.linspace(min(X), max(X), 1000)
y_linear = linear_func(x_curve, *popt_linear)
y_quadratic = quadratic_func(x_curve, *popt_quadratic)
y_cubic = cubic_func(x_curve, *popt_cubic)

# Calculate R-squared (coefficient of determination) to evaluate goodness of fit
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Evaluate R-squared for each fit
r2_linear = r_squared(Y, linear_func(X, *popt_linear))
r2_quadratic = r_squared(Y, quadratic_func(X, *popt_quadratic))
r2_cubic = r_squared(Y, cubic_func(X, *popt_cubic))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Data')
plt.plot(x_curve, y_linear, label=f'Linear Fit (R²={r2_linear:.4f})')
plt.plot(x_curve, y_quadratic, label=f'Quadratic Fit (R²={r2_quadratic:.4f})')
plt.plot(x_curve, y_cubic, label=f'Cubic Fit (R²={r2_cubic:.4f})')
plt.xlabel('Pixel Displacement (px)')
plt.ylabel('Object Height (mm)')
plt.title('Pixel Displacement vs Object Height')
plt.legend()
plt.grid(True)
plt.show()
