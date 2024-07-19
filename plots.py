import matplotlib.pyplot as plt

# Data from the table
pixel_displacement = [24.84514268, 55.76918981, 57.79133424, 90.29026038]
measured_height = [3.68, 8.94, 9.36, 14]
calculated_height = [4.717, 9.8647, 10.1767, 14.8384]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(pixel_displacement, measured_height, label='Measured Height (mm)', marker='o')
plt.plot(pixel_displacement, calculated_height, label='Calculated Height (mm)', marker='x')

# Add labels and title
plt.xlabel('Pixel Displacement')
plt.ylabel('Height (mm)')
plt.title('Height vs Pixel Displacement')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
