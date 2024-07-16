import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Resets the color to default

# Given y-values
y_values = [387, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 386, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 385, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 382, 381, 381, 381, 
381, 382, 382, 382, 382, 381, 382, 239, 239, 239, 239, 240, 239, 239, 239, 238, 239, 238, 238, 238, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 238, 238, 238, 238, 238, 238, 238, 238, 237, 237, 237, 237, 237, 237, 236, 236, 236, 236, 236, 236, 236, 236, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 233, 233, 237, 233, 233, 233, 233, 233, 233, 233, 232, 232, 237, 237, 237, 237, 237, 237, 237, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 230, 230, 230, 230, 230, 230, 230, 235, 235, 235, 235, 235, 235, 235, 230, 230, 230, 230, 230, 230, 230, 230, 230, 234, 234, 234, 
230, 230, 230, 230, 230, 230, 230, 230, 230, 233, 233, 233, 233, 230, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 230, 230, 230, 231, 231, 231, 231, 230, 230, 230, 230, 230, 230, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 230, 230, 231, 231, 231, 231, 231, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 231, 373, 373, 373, 372, 372, 371, 371, 371, 371, 370, 370, 370, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 
371, 371, 371, 371, 371, 371, 371, 371, 371, 371, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370]
print(len(y_values))
# Define the x-values (indices)
x_values = np.arange(len(y_values))


# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Initial guess for the parameters: amplitude, mean, stddev
initial_guess = [max(y_values), np.argmax(y_values), 1]

# Perform the curve fitting
try:
    params, covariance = curve_fit(gaussian, x_values, y_values, p0=initial_guess, maxfev=20000)
except RuntimeError as e:
    print(f"{Colors.FAIL}Curve fitting failed: {e}{Colors.ENDC}")
    print(np.argmax(y_values))

# Extract the fitted parameters
fitted_amplitude, fitted_mean, fitted_stddev = params

# Find the index closest to the mean
closest_index = np.argmin(np.abs(x_values - fitted_mean))
print("Closest index to the mean:", closest_index)

# Plot the original data and the fitted Gaussian curve
plt.plot(x_values, y_values, 'b-', label='data')
plt.plot(x_values, gaussian(x_values, *params), 'r--', label='fit: mean=%5.3f' % fitted_mean)
plt.axvline(x=fitted_mean, color='g', linestyle='--', label='Mean (µ)')
plt.axvline(x=closest_index, color='m', linestyle='--', label='Closest Index')
plt.xlabel('Index')
plt.ylabel('Y-values')
plt.legend()
plt.show()

closest_index, fitted_mean
