import json
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt

# Function to load kinematics data from a JSON file
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return np.array([frame['joint_velocities'] for frame in data['kinematics_frames']])

# Normalize data
def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Load the data
data1 = load_data('C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/files/combined_data_combined_subset2.json')
data2 = load_data('C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/files/combined_data_combined_subset3.json')

# Normalize the data
data1 = normalize(data1)
data2 = normalize(data2)

# Compute the DTW distance
distance, paths = dtw.warping_paths(data1.flatten(), data2.flatten(), window=25, psi=2)  # window and psi parameters can be adjusted
print("DTW Distance between the two walking sequences:", distance)

# Extract the best path
best_path = dtw.best_path(paths)

# Visualizing the warping path
plt.figure()
plt.plot([p[0] for p in best_path], [p[1] for p in best_path])
plt.title('DTW Warping Path')
plt.xlabel('Index in Sequence 1')
plt.ylabel('Index in Sequence 2')
plt.grid(True)
plt.show()
