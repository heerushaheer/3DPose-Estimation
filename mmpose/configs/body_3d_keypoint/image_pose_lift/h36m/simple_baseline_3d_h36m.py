from mmcv import Config
from mmpose.models import PoseLifter


# Model settings
model = dict(
    type='PoseLifter',
    #pretrained=None,
    backbone=dict(type='SimpleBaseline3D', in_channels=2, num_joints=17, base_channels=1024),
    #keypoint_head=dict(
        #type='PoseLifterHead',
        #in_channels=1024,
        #out_channels=3,
        #num_joints=17,
        #loss_keypoint=dict(type='MPJPELoss'),
    ),


# Dataset settings
data_root = 'data/h36m'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', flip_prob=0.5),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToTensor'),
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='Human36MDataset',
        ann_file=f'{data_root}/annotations/h36m_train.json',
        img_prefix=f'{data_root}/images/',
        pipeline=train_pipeline,
    ),
    val=dict(
        type='Human36MDataset',
        ann_file=f'{data_root}/annotations/h36m_val.json',
        img_prefix=f'{data_root}/images/',
        pipeline=train_pipeline,
    ),
)

# Training settings
optimizer = dict(type='Adam', lr=1e-3, weight_decay=1e-4)
scheduler = dict(type='StepLR', step_size=10, gamma=0.1)
total_epochs = 50
