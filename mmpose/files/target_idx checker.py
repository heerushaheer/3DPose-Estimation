import numpy as np

# Load dataset
train_path = r"C:\Users\MYSEL\mmpose\data\h3.6m\h3.6m\dataset\train_annotations_updated.npz"
data = np.load(train_path, allow_pickle=True)

# Check available keys
print("Keys in dataset:", data.keys())

# Verify `target_idx`
if 'target_idx' in data:
    print("✅ 'target_idx' exists in dataset!")
    print("First 10 target_idx values:", data['target_idx'][:10])
    print("Min target_idx:", np.min(data['target_idx']))
    print("Max target_idx:", np.max(data['target_idx']))
else:
    print("❌ 'target_idx' is missing! Dataset is incomplete.")
