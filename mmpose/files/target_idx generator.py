import numpy as np
import os

# Define dataset directory (Downloads folder)
save_dir = os.path.expanduser("~/Downloads")
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Define dataset structure
num_samples = 16000  # Adjust based on dataset size
num_joints = 133  # Number of keypoints in Human3.6M
num_coords = 3  # (x, y, z)

# Generate synthetic keypoints (3D)
keypoints_3d = np.random.randn(num_samples, num_joints, num_coords).astype(np.float32)

# Add visibility flag
vis_flag = np.ones((num_samples, num_joints, 1), dtype=np.float32)
keypoints_3d = np.concatenate((keypoints_3d, vis_flag), axis=2)  # Shape: (N, 133, 4)

# Generate correct image names for Human3.6M
subjects = ["S1", "S5", "S6", "S7", "S8"]
actions = ["Walking", "Sitting", "Running"]

imgname = np.array([
    f"{np.random.choice(subjects)}_{np.random.choice(actions)}_1.54138969_{i:06d}.jpg"
    for i in range(num_samples)
])

# Generate valid target indices (ensuring they exist in the dataset)
target_idx = np.random.randint(0, num_joints, size=num_samples)

# Define file paths in Downloads
train_path = os.path.join(save_dir, "train_annotations_updated.npz")
val_path = os.path.join(save_dir, "val_annotations_updated.npz")

# Save the `.npz` files in Downloads
np.savez(train_path, S=keypoints_3d, imgname=imgname, target_idx=target_idx)
np.savez(val_path, S=keypoints_3d[:2000], imgname=imgname[:2000], target_idx=target_idx[:2000])

print(f"✅ New annotation files saved successfully in Downloads:\n  - {train_path}\n  - {val_path}")
