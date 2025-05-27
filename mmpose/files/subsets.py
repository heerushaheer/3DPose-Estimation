import json

# Function to split and merge data into subsets
def split_and_merge_data(time_series_data, kinematics_data, frames_per_subset, file_prefix):
    subsets = []
    start = 0
    for i in range(3):  # First three subsets of 100 frames each
        end = start + frames_per_subset
        subset = {
            'time_series_frames': time_series_data[start:end],
            'kinematics_frames': kinematics_data[start:end]
        }
        subsets.append(subset)
        start = end

    # Last subset for the remaining frames
    subset = {
        'time_series_frames': time_series_data[start:],
        'kinematics_frames': kinematics_data[start:]
    }
    subsets.append(subset)

    # Save each combined subset into a separate JSON file
    for index, subset in enumerate(subsets):
        with open(f'{file_prefix}_combined_subset{index+1}.json', 'w') as file:
            json.dump(subset, file, indent=4)

# Load and check time series data structure
with open('C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/yoga_video_result_time_series_3d.json', 'r') as file:
    time_series_data = json.load(file)['all_pose_3d']  # Accessing 'all_pose_3d' directly

# Load and check kinematics data structure
with open('C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/yoga_video_result_kinematics.json', 'r') as file:
    kinematics_data = json.load(file)['frames']  # Accessing 'frames' directly

# Process and save the combined data
split_and_merge_data(time_series_data, kinematics_data, 100, 'combined_data')

print("Combined subsets created and saved successfully.")
