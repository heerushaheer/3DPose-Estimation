_base_ = ['../../../_base_/default_runtime.py']
from typing import Tuple, List  # ✅ Add this import

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=160, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.975, end=80, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=1024)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
codec = dict(
    type='VideoPoseLifting',
    num_keypoints=133,  # ✅ Updated to match dataset
    zero_center=True,
    root_index=0,
    remove_root=False)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='TCN',
        in_channels=2 * 133,  # ✅ Updated to match keypoints
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        dropout=0.25,
        use_stride_conv=True,
    ),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=133,  # ✅ Updated to match dataset
        loss=dict(type='MPJPELoss'),
        decoder=codec,
    ))

# base dataset settings
dataset_type = 'Human36mDataset'
data_root = r'C:\Users\MYSEL\mmpose\data\h3.6m\dataset'  # ✅ Updated to Windows format

def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
    instance_list, image_list = super()._load_annotations()
    h36m_data = self.ann_data

    if 'S' not in h36m_data:
        raise KeyError("❌ Missing 'S' key in dataset file. Check the annotation file path!")

    keypoints_3d = h36m_data['S']

    print(f"🔍 Debug: keypoints_3d shape = {keypoints_3d.shape}")

    if keypoints_3d.shape[2] == 3:
        vis_flag = np.ones((keypoints_3d.shape[0], keypoints_3d.shape[1], 1), dtype=np.float32)
        keypoints_3d = np.concatenate((keypoints_3d, vis_flag), axis=2)

    print(f"✅ Fixed keypoints_3d shape = {keypoints_3d.shape}")

    valid_instances = []
    skipped_instances = 0

    for idx, instance in enumerate(instance_list):
        target_idx = instance.get('target_idx', None)

        if target_idx is None or not (0 <= target_idx < keypoints_3d.shape[1]):
            print(f"⚠️ Skipping index {idx}: target_idx={target_idx} is invalid")
            skipped_instances += 1
            continue

        instance['lifting_target'] = keypoints_3d[:, target_idx, :]
        valid_instances.append(instance)

    print(f"✅ Loaded {len(valid_instances)} valid instances, Skipped {skipped_instances} instances.")

    return valid_instances, image_list

# pipelines
train_pipeline = [
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(),
        target_flip_cfg=dict(),
    ),
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]
val_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=r'C:\Users\MYSEL\mmpose\data\h3.6m\h3.6m\dataset\train_annotations_updated.npz',  # ✅ Windows path
        seq_len=27,
        causal=False,
        pad_video_seq=True,
        camera_param_file=None,
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file=r'C:\Users\MYSEL\mmpose\data\h3.6m\h3.6m\dataset\val_annotations_updated.npz',  # ✅ Windows path
        seq_len=27,
        causal=False,
        pad_video_seq=True,
        camera_param_file=None,  # ✅ Remove if missing
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe'),
    dict(type='MPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,  # ✅ Keeps benchmark settings
    dist_cfg=dict(backend='gloo')  # ✅ Only keeps distributed backend
)

