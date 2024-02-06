import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

# Load the JSON file
file_path = r"C:\Users\31602\Desktop\prompt.json"
with open(file_path, 'r') as file:
    latest_data = json.load(file)

# Counting the frequency of each value
value_counts = Counter(latest_data.values())

# Sorting the values based on their frequency (descending order)
sorted_values_by_frequency = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

# Extracting the values and their frequencies
values_sorted, frequencies_sorted = zip(*sorted_values_by_frequency)

filtered_values_0_100 = {k: v for k, v in value_counts.items() if 0 <= k <= 100}
# Sorting the filtered values based on their frequency (descending order)
sorted_filtered_values_0_100_by_frequency = sorted(filtered_values_0_100.items(), key=lambda x: x[1], reverse=True)
# Extracting the values and their frequencies for the filtered data
values_sorted_0_100, frequencies_sorted_0_100 = zip(*sorted_filtered_values_0_100_by_frequency)

sorted_0_100_data = sorted(filtered_values_0_100.items())
x_0_100, y_0_100 = zip(*sorted_0_100_data)

# Preparing data for smooth curve
x_smooth = np.linspace(min(x_0_100), max(x_0_100), 300)
spl = make_interp_spline(x_0_100, y_0_100, k=3)  # Spline interpolation of degree 3
y_smooth = spl(x_smooth)

# Creating the plot with a smooth curve and markers for data points
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid", palette="Spectral")
plt.plot(x_smooth, y_smooth, color='darkblue', label='Frequency', linewidth=1.5)  # Smooth curve

''' 
marker符号说明：
    1. 三角形可以使用'^'（向上的三角形）、'v'（向下的三角形）、'<'（向左的三角形）或'>'（向右的三角形）。
    2. 星号可以使用'*'。
'''
plt.scatter(x_0_100, y_0_100, color='red', label='Data Points', s=5, marker='^')  # Data points

# Enhancing the plot aesthetics
plt.title('Frequency Distribution of Values (0-100, Smooth Curve with Markers)', fontsize=16)
plt.xlabel('Values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend(title='Legend')
plt.show()
