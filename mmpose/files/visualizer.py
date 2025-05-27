import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#########################################
# 1) LOAD THE 3D KEYPOINTS FROM JSON
#########################################
json_path = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/result_3d.json"  # <-- Replace with your file
with open(json_path, "r") as f:
    data = json.load(f)

# Extract array: shape (T, N, 3)
all_keypoints_3d = np.array(data["all_keypoints_3d"])  
# T = number of frames, N = number of joints

print("all_keypoints_3d shape:", all_keypoints_3d.shape)  
# Example: (100, 17, 3)

#########################################
# 2) DEFINE YOUR SKELETON CONNECTIONS
#########################################
# Example for a 17-joint model
skeleton_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Spine
    (1, 5), (5, 6), (6, 7),             # Left Arm
    (1, 8), (8, 9), (9, 10),            # Right Arm
    (2, 11), (11, 12), (12, 13),        # Left Leg
    (2, 14), (14, 15), (15, 16)         # Right Leg
]

# Simple color scheme for body parts
joint_colors = {
    "spine": "blue",
    "arms":  "green",
    "legs":  "red"
}

def plot_skeleton(ax, keypoints):
    """
    Plot a single 3D skeleton frame with color-coded body parts.
    keypoints: shape (N, 3)
    """
    for (i, j) in skeleton_pairs:
        # Decide color based on joint index
        if i in [1, 2, 3, 4]:           # Spine
            color = joint_colors["spine"]
        elif i in [5, 6, 7, 8, 9, 10]:  # Arms
            color = joint_colors["arms"]
        else:                           # Legs
            color = joint_colors["legs"]

        ax.plot(
            [keypoints[i, 0], keypoints[j, 0]],
            [keypoints[i, 1], keypoints[j, 1]],
            [keypoints[i, 2], keypoints[j, 2]],
            color=color, linewidth=2, marker='o'
        )

#########################################
# 3) CREATE FIGURE & ANIMATION
#########################################
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Skeleton Animation", fontsize=14)

# Pre-set axis limits & labels
ax.set_xlim(-60000, 60000)  # Adjust based on your data
ax.set_ylim(-60000, 60000)
ax.set_zlim(-30000, 80000)
ax.set_xlabel("X", fontsize=10)
ax.set_ylabel("Y", fontsize=10)
ax.set_zlabel("Z", fontsize=10)

# We'll store the total number of frames & assume we want real-time speed
T = all_keypoints_3d.shape[0]
fps = 30  # or your video FPS

# Dummy scatter handle for returning in update
scatter_handle, = ax.plot([], [], [], 'bo', markersize=5)

def update(frame):
    """
    Called each frame to update the skeleton.
    """
    # Clear previous skeleton
    ax.cla()

    # Reset axis
    ax.set_xlim(-60000, 60000)
    ax.set_ylim(-60000, 60000)
    ax.set_zlim(-30000, 80000)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.set_title("3D Skeleton Animation", fontsize=14)

    # Plot skeleton for the current frame
    current_kpts = all_keypoints_3d[frame]  # shape (N, 3)
    plot_skeleton(ax, current_kpts)

    return scatter_handle,

ani = animation.FuncAnimation(
    fig, update, frames=T, interval=1000/fps, blit=False
)

#########################################
# 4) SAVE AS MP4 (Requires ffmpeg)
#########################################
output_path = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/3d_skeleton_animation.mp4"
ani.save(output_path, writer="ffmpeg", fps=fps)
print(f"Animation saved to: {output_path}")

# Optional: Show interactive window
# plt.show()
