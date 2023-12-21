import matplotlib.pyplot as plt
import numpy as np

# Generate random data for three groups
np.random.seed(42)  # Setting seed for reproducibility
data_group1 = np.random.normal(loc=0, scale=1, size=100)
data_group2 = np.random.normal(loc=2, scale=1, size=100)
data_group3 = np.random.normal(loc=-2, scale=1, size=100)

# Combine the data into a list
data = [data_group1, data_group2, data_group3]

# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'], notch=True, patch_artist=True)

# Customize the plot
plt.title('Box Plot of Random Data')
plt.xlabel('Groups')
plt.ylabel('Values')

# Show the plot
plt.show()
