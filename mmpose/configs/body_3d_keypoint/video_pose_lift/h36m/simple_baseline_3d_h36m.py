_base_ = ['configs7body_3d_keypoint/video_pose_lift/h36m/simple_baseline_3d_h36m.py']

data = dict(
    test=dict(
        type='Body3DH36MDataset',
        ann_file='C:/Users/yayr24/OneDrive - BTH Student/Desktop/mmpose/formatted_keypoints.json',
        img_prefix='',
        data_cfg=dict(
            num_joints=17,
            image_size=[1920, 1080],  # Adjust based on your dataset
        ),
        pipeline=[  # Data loading and processing pipeline
            dict(type='LoadImageFromFile'),
            dict(type='GetBBoxCenterScale'),
            dict(type='ResizeTransform'),
            dict(type='GetKeypoints')
        ],
    ),
)
